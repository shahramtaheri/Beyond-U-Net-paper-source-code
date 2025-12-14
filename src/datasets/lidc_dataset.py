# src/datasets/lidc_dataset.py
# 2D slice dataset for LIDC-IDRI (or any image/mask pairs).
# Expects fold JSON items shaped like:
#   {"image": "path/to/img.png", "mask": "path/to/mask.png", "id": "optional_unique_id"}
#
# Returns:
#   {"image": FloatTensor[C,H,W], "mask": FloatTensor[1,H,W], "id": str}
#
# Supports:
# - image/mask formats: .png/.jpg/.jpeg/.tif/.tiff and .npy
# - optional Albumentations augmentations for training
# - optional intensity normalization (assumes preprocessed to [0,1] already; keeps safe fallback)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image


def _read_array(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()

    # NPY
    if ext == ".npy":
        arr = np.load(str(p))
        return arr

    # Common image formats
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        img = Image.open(str(p))
        arr = np.array(img)
        return arr

    raise ValueError(f"Unsupported file extension: {ext} for path {p}")


def _ensure_hw(arr: np.ndarray) -> np.ndarray:
    # Ensure shape is HxW (single-channel). If RGB, convert to grayscale by taking channel 0.
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # If HxWxC or CxHxW, try to infer
        if arr.shape[-1] in (1, 3, 4):  # HWC
            return arr[..., 0]
        if arr.shape[0] in (1, 3, 4):   # CHW
            return arr[0, ...]
    raise ValueError(f"Expected 2D array or image-like 3D array, got shape {arr.shape}")


def _to_float01(img: np.ndarray) -> np.ndarray:
    # If already float and in [0,1], keep. Else scale reasonably.
    if img.dtype.kind in {"f"}:
        # clip to [0,1] to be safe
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    # uint8 images: scale to [0,1]
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0)

    # other integer types: min-max scale (safe fallback)
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - mn) / (mx - mn)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _to_binary_mask(msk: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if msk.dtype.kind in {"f"}:
        return (msk >= thr).astype(np.float32)
    if msk.dtype == np.uint8:
        # common convention: 0/255
        return (msk > 127).astype(np.float32)
    # integer masks: assume 0/1 or 0..k
    return (msk > 0).astype(np.float32)


def _default_id(item: Dict[str, Any], idx: int, image_key: str, mask_key: str) -> str:
    if "id" in item and item["id"] is not None:
        return str(item["id"])
    # derive from filename
    ip = Path(item[image_key]).stem if image_key in item else f"img_{idx}"
    mp = Path(item[mask_key]).stem if mask_key in item else f"msk_{idx}"
    return f"{ip}__{mp}"


@dataclass
class AugmentConfig:
    # Keep small-angle & mild deformations per your manuscript
    rotate_limit: int = 15
    p_flip: float = 0.5
    p_rotate: float = 0.5
    p_elastic: float = 0.2
    p_noise: float = 0.2


class LIDCSliceDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        image_key: str = "image",
        mask_key: str = "mask",
        augment: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.items = items
        self.image_key = image_key
        self.mask_key = mask_key
        self.augment = augment
        self.cfg = cfg or {}

        self._alb = None
        self._use_alb = False

        # Set up Albumentations transforms if available and requested
        if augment:
            try:
                import albumentations as A

                acfg = self.cfg.get("augmentation", {})
                aug = AugmentConfig(
                    rotate_limit=int(acfg.get("rotate_limit", 15)),
                    p_flip=float(acfg.get("p_flip", 0.5)),
                    p_rotate=float(acfg.get("p_rotate", 0.5)),
                    p_elastic=float(acfg.get("p_elastic", 0.2)),
                    p_noise=float(acfg.get("p_noise", 0.2)),
                )

                self._alb = A.Compose(
                    [
                        A.HorizontalFlip(p=aug.p_flip),
                        A.VerticalFlip(p=aug.p_flip),
                        A.Rotate(limit=aug.rotate_limit, border_mode=0, p=aug.p_rotate),
                        A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=10, border_mode=0, p=aug.p_elastic),
                        A.GaussNoise(var_limit=(5.0, 25.0), p=aug.p_noise),
                    ],
                    additional_targets={"mask": "mask"},
                )
                self._use_alb = True
            except Exception:
                # Albumentations not installed; proceed without augmentation
                self._alb = None
                self._use_alb = False

        # Optional: verify keys exist
        for it in self.items[: min(3, len(self.items))]:
            if self.image_key not in it or self.mask_key not in it:
                raise KeyError(f"Each item must contain '{self.image_key}' and '{self.mask_key}'. Got keys: {list(it.keys())}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        img_path = item[self.image_key]
        msk_path = item[self.mask_key]

        img = _read_array(img_path)
        msk = _read_array(msk_path)

        img = _ensure_hw(img)
        msk = _ensure_hw(msk)

        img = _to_float01(img)
        msk = _to_binary_mask(msk)

        # Albumentations expects HxW and mask HxW
        if self._use_alb and self._alb is not None:
            out = self._alb(image=img, mask=msk)
            img = out["image"]
            msk = out["mask"]

        # Convert to torch tensors: image -> (1,H,W), mask -> (1,H,W)
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        msk_t = torch.from_numpy(msk).float().unsqueeze(0)

        sample_id = _default_id(item, idx, self.image_key, self.mask_key)

        return {"image": img_t, "mask": msk_t, "id": sample_id}
