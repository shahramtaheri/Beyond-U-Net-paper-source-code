# src/evaluate.py
# Evaluation script for lung nodule segmentation (2D).
# - Loads a trained checkpoint
# - Evaluates on a chosen split (val/test) from fold JSON
# - Reports: DSC, IoU, Sensitivity, PPV, HD, HD95, ASSD
# - Saves per-case metrics + aggregate summary CSV/JSON

from __future__ import annotations

import json
import yaml
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.ndimage import binary_erosion, distance_transform_edt


# -------------------------
# Optional: import your local modules
# -------------------------
try:
    from src.models.residual_unetpp_attention import ResidualAttentionUNetPP
except Exception:
    ResidualAttentionUNetPP = None  # type: ignore

try:
    from src.datasets.lidc_dataset import LIDCSliceDataset
except Exception:
    LIDCSliceDataset = None  # type: ignore


# -------------------------
# Helpers
# -------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int, deterministic: bool = True) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(cfg: Dict[str, Any]) -> nn.Module:
    mcfg = cfg["model"]
    name = str(mcfg.get("name", "ResidualAttentionUNetPP")).lower()
    in_ch = int(mcfg.get("in_channels", 1))
    out_ch = int(mcfg.get("out_channels", 1))
    filters = mcfg.get("filters", [64, 128, 256, 512, 1024])

    if name in {"residualattentionunetpp", "residual_attention_unetpp", "unetpp_res_att"}:
        if ResidualAttentionUNetPP is None:
            raise ImportError("Could not import ResidualAttentionUNetPP. Check src/models/residual_unetpp_attention.py")
        return ResidualAttentionUNetPP(in_ch=in_ch, out_ch=out_ch, filters=tuple(filters))
    raise ValueError(f"Unknown model name: {mcfg.get('name')}")

def load_checkpoint(model: nn.Module, ckpt_path: str | Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    # Supports either {"model_state": ...} or raw state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif isinstance(ckpt, dict):
        # Might be a plain state_dict
        model.load_state_dict(ckpt, strict=True)
    else:
        raise ValueError("Unsupported checkpoint format.")
    model.to(device)
    model.eval()
    return ckpt if isinstance(ckpt, dict) else {}

def sigmoid_to_mask(logits: torch.Tensor, thr: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= thr).float()

def _to_numpy_mask(x: torch.Tensor) -> np.ndarray:
    # expects shape (1,H,W) or (H,W)
    x = x.detach().float().cpu()
    if x.ndim == 3:
        x = x[0]
    return (x.numpy() > 0.5).astype(np.bool_)

def dice_iou_sens_ppv(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Tuple[float, float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    denom_dice = (pred.sum() + gt.sum())
    dice = (2.0 * tp + eps) / (denom_dice + eps)

    denom_iou = (tp + fp + fn)
    iou = (tp + eps) / (denom_iou + eps)

    sens = (tp + eps) / (tp + fn + eps)
    ppv = (tp + eps) / (tp + fp + eps)

    return float(dice), float(iou), float(sens), float(ppv)

def surface_distances_2d(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute symmetric surface-to-surface distances between binary masks a and b (2D).
    Returns concatenated distances (a->b and b->a) in pixels.
    If one mask is empty and the other is not, returns None (undefined boundary metrics).
    If both empty, returns empty array (all boundary metrics = 0 by convention).
    """
    a = a.astype(bool)
    b = b.astype(bool)

    if a.sum() == 0 and b.sum() == 0:
        return np.array([], dtype=np.float32)

    if a.sum() == 0 or b.sum() == 0:
        return None

    # 2D surfaces: boundary = mask XOR eroded(mask)
    struct = np.ones((3, 3), dtype=bool)
    a_er = binary_erosion(a, structure=struct, border_value=0)
    b_er = binary_erosion(b, structure=struct, border_value=0)

    a_surf = np.logical_xor(a, a_er)
    b_surf = np.logical_xor(b, b_er)

    if a_surf.sum() == 0 and b_surf.sum() == 0:
        return np.array([], dtype=np.float32)
    if a_surf.sum() == 0 or b_surf.sum() == 0:
        return None

    # distance transform of inverse surface maps each pixel to nearest surface pixel
    dt_b = distance_transform_edt(np.logical_not(b_surf))
    dt_a = distance_transform_edt(np.logical_not(a_surf))

    d_ab = dt_b[a_surf]
    d_ba = dt_a[b_surf]
    return np.concatenate([d_ab, d_ba]).astype(np.float32)

def hd_hd95_assd(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """
    Hausdorff Distance (HD), 95th percentile Hausdorff (HD95), ASSD for 2D masks.
    Returns (HD, HD95, ASSD). If undefined (one empty, one non-empty) returns (nan,nan,nan).
    If both empty: (0,0,0).
    """
    d = surface_distances_2d(a, b)
    if d is None:
        return (float("nan"), float("nan"), float("nan"))
    if d.size == 0:
        return (0.0, 0.0, 0.0)

    hd = float(np.max(d))
    hd95 = float(np.percentile(d, 95))
    assd = float(np.mean(d))
    return (hd, hd95, assd)

def build_loader_for_split(cfg: Dict[str, Any], split: str) -> DataLoader:
    """
    Expects cfg["data"]["splits"]["fold_json"] points to a JSON containing keys:
      - "train": [...]
      - "val":   [...]
      - "test":  [...]   (optional)
    Each item should at least include paths for image & mask.
    """
    if LIDCSliceDataset is None:
        raise ImportError("Could not import LIDCSliceDataset. Check src/datasets/lidc_dataset.py")

    dcfg = cfg["data"]
    split_path = Path(dcfg["splits"]["fold_json"])
    with open(split_path, "r", encoding="utf-8") as f:
        fold = json.load(f)

    if split not in fold:
        if split == "test":
            raise KeyError(f"'{split}' split not found in {split_path}. Available keys: {list(fold.keys())}")
        raise KeyError(f"'{split}' split not found in {split_path}.")

    items = fold[split]
    img_key = dcfg.get("img_key", "image")
    msk_key = dcfg.get("mask_key", "mask")

    ds = LIDCSliceDataset(
        items=items,
        image_key=img_key,
        mask_key=msk_key,
        augment=False,
        cfg=cfg,
    )

    tcfg = cfg["train"]
    num_workers = int(tcfg.get("num_workers", 4))
    batch_size = int(tcfg.get("batch_size", 8))
    pin_memory = bool(tcfg.get("pin_memory", True))

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

@dataclass
class EvalRow:
    case_id: str
    dice: float
    iou: float
    sens: float
    ppv: float
    hd: float
    hd95: float
    assd: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config (fold-specific).")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint (.pt).")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument("--thr", default=0.5, type=float, help="Sigmoid threshold for binarization.")
    parser.add_argument("--outdir", default=None, type=str, help="Override output directory.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Reproducibility
    seed_cfg = cfg.get("seed", {})
    set_seed(int(seed_cfg.get("seed", 42)), deterministic=bool(seed_cfg.get("deterministic", True)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir) if args.outdir else Path(cfg["train"]["outdir"]) / "eval"
    outdir = ensure_dir(outdir)

    # Build + load
    model = build_model(cfg)
    ckpt_meta = load_checkpoint(model, args.ckpt, device)
    nparams = count_params(model)

    loader = build_loader_for_split(cfg, args.split)

    rows: List[EvalRow] = []
    thr = float(args.thr)

    # Track validity of boundary metrics (HD, HD95, ASSD)
    boundary_valid = 0
    boundary_total = 0

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            logits = model(x)
            pred = sigmoid_to_mask(logits, thr=thr)

            # Per-sample metrics
            for i in range(x.size(0)):
                # Case id (best effort)
                case_id = None
                if "id" in batch:
                    case_id = str(batch["id"][i])
                elif "case_id" in batch:
                    case_id = str(batch["case_id"][i])
                elif "meta" in batch and isinstance(batch["meta"], list) and i < len(batch["meta"]):
                    m = batch["meta"][i]
                    if isinstance(m, dict) and "id" in m:
                        case_id = str(m["id"])
                if case_id is None:
                    case_id = f"{args.split}_{bi:04d}_{i:02d}"

                pred_np = _to_numpy_mask(pred[i])
                gt_np = _to_numpy_mask(y[i])

                dsc, iou, sens, ppv = dice_iou_sens_ppv(pred_np, gt_np)

                hd, hd95, assd = hd_hd95_assd(pred_np, gt_np)
                boundary_total += 1
                if np.isfinite(hd) and np.isfinite(hd95) and np.isfinite(assd):
                    boundary_valid += 1

                rows.append(EvalRow(case_id, dsc, iou, sens, ppv, hd, hd95, assd))

    # Aggregate statistics (mean over cases; boundary metrics mean over valid cases)
    dice_vals = np.array([r.dice for r in rows], dtype=np.float32)
    iou_vals = np.array([r.iou for r in rows], dtype=np.float32)
    sens_vals = np.array([r.sens for r in rows], dtype=np.float32)
    ppv_vals = np.array([r.ppv for r in rows], dtype=np.float32)

    hd_vals = np.array([r.hd for r in rows], dtype=np.float32)
    hd95_vals = np.array([r.hd95 for r in rows], dtype=np.float32)
    assd_vals = np.array([r.assd for r in rows], dtype=np.float32)

    summary = {
        "split": args.split,
        "threshold": thr,
        "num_cases": int(len(rows)),
        "model_params_million": float(nparams / 1e6),
        "dice_mean": float(np.mean(dice_vals)) if len(dice_vals) else float("nan"),
        "iou_mean": float(np.mean(iou_vals)) if len(iou_vals) else float("nan"),
        "sensitivity_mean": float(np.mean(sens_vals)) if len(sens_vals) else float("nan"),
        "ppv_mean": float(np.mean(ppv_vals)) if len(ppv_vals) else float("nan"),
        "hd_mean": float(np.nanmean(hd_vals)) if np.any(np.isfinite(hd_vals)) else float("nan"),
        "hd95_mean": float(np.nanmean(hd95_vals)) if np.any(np.isfinite(hd95_vals)) else float("nan"),
        "assd_mean": float(np.nanmean(assd_vals)) if np.any(np.isfinite(assd_vals)) else float("nan"),
        "boundary_metrics_valid_cases": int(boundary_valid),
        "boundary_metrics_total_cases": int(boundary_total),
        "ckpt_path": str(args.ckpt),
    }

    # Save per-case CSV
    per_case_csv = outdir / f"per_case_{args.split}.csv"
    with open(per_case_csv, "w", encoding="utf-8") as f:
        f.write("case_id,dice,iou,sensitivity,ppv,hd,hd95,assd\n")
        for r in rows:
            f.write(
                f"{r.case_id},{r.dice:.6f},{r.iou:.6f},{r.sens:.6f},{r.ppv:.6f},"
                f"{r.hd:.6f},{r.hd95:.6f},{r.assd:.6f}\n"
            )

    # Save summary JSON
    summary_json = outdir / f"summary_{args.split}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("Evaluation complete")
    print(f"Split: {args.split} | thr={thr}")
    print(f"Cases: {summary['num_cases']} | Params: {summary['model_params_million']:.3f} M")
    print(f"DSC:  {summary['dice_mean']:.4f}")
    print(f"IoU:  {summary['iou_mean']:.4f}")
    print(f"Sens: {summary['sensitivity_mean']:.4f}")
    print(f"PPV:  {summary['ppv_mean']:.4f}")
    print(f"HD:   {summary['hd_mean']:.4f} (valid {boundary_valid}/{boundary_total})")
    print(f"HD95: {summary['hd95_mean']:.4f} (valid {boundary_valid}/{boundary_total})")
    print(f"ASSD: {summary['assd_mean']:.4f} (valid {boundary_valid}/{boundary_total})")
    print(f"Saved per-case CSV: {per_case_csv}")
    print(f"Saved summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
