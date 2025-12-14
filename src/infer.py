# src/infer.py
# Inference script for 2D lung nodule segmentation.
# - Loads a checkpoint
# - Runs inference on a single image file or a directory of images
# - Saves predicted masks (PNG) and optional overlay images
#
# Assumptions:
# - Input images are already preprocessed similarly to training (e.g., [0,1] range).
# - Supports .png/.jpg/.jpeg/.tif/.tiff and .npy for images.
# - Model outputs logits; sigmoid + threshold used for mask.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from src.models.residual_unetpp_attention import ResidualAttentionUNetPP
except Exception:
    ResidualAttentionUNetPP = None  # type: ignore


# -------------------------
# I/O helpers
# -------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    ext = p.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(p))
        return arr
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        img = Image.open(str(p))
        return np.array(img)
    raise ValueError(f"Unsupported image extension: {ext}")


def to_gray_hw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # HWC or CHW
        if arr.shape[-1] in (1, 3, 4):
            return arr[..., 0]
        if arr.shape[0] in (1, 3, 4):
            return arr[0, ...]
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def to_float01(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) / 255.0)
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    arr = (arr - mn) / (mx - mn)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def save_png_mask(mask01: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    mask_u8 = (np.clip(mask01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(mask_u8).save(str(out_path))


def save_overlay(img01: np.ndarray, mask01: np.ndarray, out_path: str | Path, alpha: float = 0.4) -> None:
    """
    Creates a simple red overlay for the predicted mask.
    img01: HxW float in [0,1]
    mask01: HxW float in [0,1]
    """
    out_path = Path(out_path)
    base = np.clip(img01, 0.0, 1.0)
    m = (mask01 >= 0.5).astype(np.float32)

    rgb = np.stack([base, base, base], axis=-1)  # grayscale -> RGB
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0  # red channel

    overlay = rgb * (1.0 - alpha * m[..., None]) + red * (alpha * m[..., None])
    overlay_u8 = (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(overlay_u8).save(str(out_path))


# -------------------------
# Model helpers
# -------------------------

def build_model(filters=(64, 128, 256, 512, 1024), in_ch: int = 1, out_ch: int = 1) -> nn.Module:
    if ResidualAttentionUNetPP is None:
        raise ImportError("Could not import ResidualAttentionUNetPP. Check src/models/residual_unetpp_attention.py")
    return ResidualAttentionUNetPP(in_ch=in_ch, out_ch=out_ch, filters=tuple(filters))


def load_checkpoint(model: nn.Module, ckpt_path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=True)
    else:
        raise ValueError("Unsupported checkpoint format.")
    model.to(device)
    model.eval()


@torch.no_grad()
def predict_mask(model: nn.Module, img01_hw: np.ndarray, device: torch.device, thr: float = 0.5) -> np.ndarray:
    """
    img01_hw: HxW float in [0,1]
    returns: HxW float mask in {0,1}
    """
    x = torch.from_numpy(img01_hw).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    x = x.to(device)
    logits = model(x)
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()
    pred_hw = pred[0, 0].detach().cpu().numpy().astype(np.float32)
    return pred_hw


def list_inputs(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    files = []
    for ext in sorted(IMG_EXTS):
        files.extend(inp.glob(f"*{ext}"))
    return sorted(set(files))


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint (.pt).")
    parser.add_argument("--input", required=True, type=str, help="Path to an image file or a directory.")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--thr", default=0.5, type=float, help="Threshold for sigmoid probability.")
    parser.add_argument("--save_overlay", action="store_true", help="Save overlay PNG images.")
    parser.add_argument("--alpha", default=0.4, type=float, help="Overlay alpha.")
    parser.add_argument("--filters", default="64,128,256,512,1024", type=str, help="Comma-separated filters.")
    args = parser.parse_args()

    outdir = ensure_dir(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filters = tuple(int(x.strip()) for x in args.filters.split(",") if x.strip())
    model = build_model(filters=filters, in_ch=1, out_ch=1)
    load_checkpoint(model, args.ckpt, device)

    inp = Path(args.input)
    paths = list_inputs(inp)
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {inp}")

    masks_dir = ensure_dir(outdir / "masks")
    overlays_dir = ensure_dir(outdir / "overlays") if args.save_overlay else None

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Inputs: {len(paths)}")
    print(f"Output: {outdir}")
    print(f"Threshold: {args.thr}")

    for p in paths:
        arr = load_image(p)
        img = to_gray_hw(arr)
        img01 = to_float01(img)

        pred = predict_mask(model, img01, device=device, thr=float(args.thr))

        stem = p.stem
        save_png_mask(pred, masks_dir / f"{stem}_pred.png")

        if overlays_dir is not None:
            save_overlay(img01, pred, overlays_dir / f"{stem}_overlay.png", alpha=float(args.alpha))

    print("Inference complete.")
    print(f"Saved masks to: {masks_dir}")
    if overlays_dir is not None:
        print(f"Saved overlays to: {overlays_dir}")


if __name__ == "__main__":
    main()
