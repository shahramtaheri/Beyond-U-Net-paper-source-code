# src/train.py
# Training script for lung nodule segmentation (2D) with 5-fold CV support.
# - YAML config driven
# - deterministic seeds
# - early stopping
# - checkpointing (best by val Dice)
# - logs to stdout + optional CSV

from __future__ import annotations

import os
import sys
import math
import time
import json
import yaml
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -------------------------
# Optional: import your local modules
# -------------------------
# Expected repository layout:
# src/
#   models/residual_unetpp_attention.py   (contains ResidualAttentionUNetPP or equivalent)
#   losses/hybrid.py                      (contains HybridDiceFocalLoss)
#   datasets/lidc_dataset.py              (contains LIDCSliceDataset)
#   utils/seed.py                         (set_seed)
#   utils/checkpoint.py                   (save_checkpoint)
#
# If your module names differ, update imports below accordingly.

try:
    from src.models.residual_unetpp_attention import ResidualAttentionUNetPP
except Exception:
    # fallback: allow running if you placed model elsewhere; user can edit
    ResidualAttentionUNetPP = None  # type: ignore

try:
    from src.losses.hybrid import HybridDiceFocalLoss
except Exception:
    HybridDiceFocalLoss = None  # type: ignore

try:
    from src.datasets.lidc_dataset import LIDCSliceDataset
except Exception:
    LIDCSliceDataset = None  # type: ignore


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-7) -> Tuple[float, float]:
    """
    Computes mean Dice and IoU over batch from logits and binary targets.
    logits: (N,1,H,W) or (N,H,W)
    targets: same shape, values 0/1
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = (preds + targets).clamp(0, 1).sum(dim=1)  # for Dice denom
    denom = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * inter + eps) / (denom + eps)
    iou = (inter + eps) / ((preds.sum(dim=1) + targets.sum(dim=1) - inter) + eps)

    return float(dice.mean().item()), float(iou.mean().item())


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@dataclass
class EarlyStopping:
    patience: int = 15
    mode: str = "max"  # "max" for Dice, "min" for loss
    min_delta: float = 0.0

    best: Optional[float] = None
    bad_epochs: int = 0

    def step(self, value: float) -> Tuple[bool, bool]:
        """
        Returns: (should_stop, is_best)
        """
        is_best = False
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            is_best = True
            return False, is_best

        improved = (value - self.best) > self.min_delta if self.mode == "max" else (self.best - value) > self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
            is_best = True
        else:
            self.bad_epochs += 1

        should_stop = self.bad_epochs >= self.patience
        return should_stop, is_best


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    mcfg = cfg["model"]
    name = mcfg.get("name", "ResidualAttentionUNetPP")

    in_ch = int(mcfg.get("in_channels", 1))
    out_ch = int(mcfg.get("out_channels", 1))
    filters = mcfg.get("filters", [64, 128, 256, 512, 1024])

    if name.lower() in {"residualattentionunetpp", "residual_attention_unetpp", "unetpp_res_att"}:
        if ResidualAttentionUNetPP is None:
            raise ImportError("Could not import ResidualAttentionUNetPP. Check src/models/residual_unetpp_attention.py")
        return ResidualAttentionUNetPP(in_ch=in_ch, out_ch=out_ch, filters=tuple(filters))
    else:
        raise ValueError(f"Unknown model name: {name}")


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    lcfg = cfg["loss"]
    name = lcfg.get("name", "HybridDiceFocalLoss")

    if name.lower() in {"hybriddicefocalloss", "dice_focal", "hybrid"}:
        if HybridDiceFocalLoss is None:
            raise ImportError("Could not import HybridDiceFocalLoss. Check src/losses/hybrid.py")
        lam = float(lcfg.get("lambda", 0.5))
        alpha = float(lcfg.get("alpha", 0.25))
        gamma = float(lcfg.get("gamma", 2.0))
        return HybridDiceFocalLoss(lam=lam, alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss name: {name}")


def build_optim(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    ocfg = cfg["optim"]
    name = ocfg.get("name", "adam").lower()
    lr = float(ocfg.get("lr", 1e-4))
    wd = float(ocfg.get("weight_decay", 1e-5))

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    scfg = cfg.get("scheduler", {"name": "reduce_on_plateau"})
    name = scfg.get("name", "reduce_on_plateau").lower()

    if name in {"reduce_on_plateau", "reducelronplateau"}:
        factor = float(scfg.get("factor", 0.5))
        patience = int(scfg.get("patience", 5))
        min_lr = float(scfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr, verbose=False
        )

    if name in {"cosine", "cosineannealing"}:
        tmax = int(scfg.get("tmax", cfg["train"]["epochs"]))
        eta_min = float(scfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)

    return None


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Expects cfg["data"]["splits"] to point to a fold JSON listing train/val items (paths).
    You can adapt this to your own split format.
    """
    if LIDCSliceDataset is None:
        raise ImportError("Could not import LIDCSliceDataset. Check src/datasets/lidc_dataset.py")

    dcfg = cfg["data"]
    split_path = Path(dcfg["splits"]["fold_json"])
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_items = split["train"]
    val_items = split["val"]

    img_key = dcfg.get("img_key", "image")
    msk_key = dcfg.get("mask_key", "mask")

    # Dataset should accept items as list[dict] with image/mask paths, plus transforms/augment flags
    train_ds = LIDCSliceDataset(
        items=train_items,
        image_key=img_key,
        mask_key=msk_key,
        augment=True,
        cfg=cfg,
    )
    val_ds = LIDCSliceDataset(
        items=val_items,
        image_key=img_key,
        mask_key=msk_key,
        augment=False,
        cfg=cfg,
    )

    tcfg = cfg["train"]
    num_workers = int(tcfg.get("num_workers", 4))
    batch_size = int(tcfg.get("batch_size", 8))
    pin_memory = bool(tcfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, val_loader


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = True,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        loss_meter.update(float(loss.item()), bs)

        d, j = dice_iou_from_logits(logits.detach(), y.detach())
        dice_meter.update(d, bs)
        iou_meter.update(j, bs)

    return loss_meter.avg, dice_meter.avg, iou_meter.avg


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = True,
) -> Tuple[float, float, float]:
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        bs = x.size(0)
        loss_meter.update(float(loss.item()), bs)
        d, j = dice_iou_from_logits(logits, y)
        dice_meter.update(d, bs)
        iou_meter.update(j, bs)

    return loss_meter.avg, dice_meter.avg, iou_meter.avg


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config (fold-specific).")
    parser.add_argument("--override_outdir", default=None, type=str, help="Optional override for output directory.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Output directory
    outdir = Path(args.override_outdir) if args.override_outdir else Path(cfg["train"]["outdir"])
    outdir = ensure_dir(outdir)

    # Save a copy of config for reproducibility
    save_yaml(cfg, outdir / "config_used.yaml")

    # Seeds / determinism
    seed_cfg = cfg.get("seed", {})
    seed = int(seed_cfg.get("seed", 42))
    deterministic = bool(seed_cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Log model info
    nparams = count_params(model)
    print(f"Device: {device}")
    print(f"Train outdir: {outdir}")
    print(f"Model params: {nparams/1e6:.3f} M")

    # Dataloaders
    train_loader, val_loader = build_loaders(cfg)

    # Training config
    tcfg = cfg["train"]
    epochs = int(tcfg.get("epochs", 100))
    amp = bool(tcfg.get("amp", True))
    grad_clip = tcfg.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    # Early stopping on validation Dice
    es = EarlyStopping(
        patience=int(tcfg.get("early_stopping_patience", 15)),
        mode="max",
        min_delta=float(tcfg.get("early_stopping_min_delta", 0.0)),
    )

    # Optional CSV log
    csv_path = outdir / "train_log.csv"
    write_csv = bool(tcfg.get("write_csv", True))
    if write_csv:
        if not csv_path.exists():
            csv_path.write_text(
                "epoch,lr,train_loss,train_dice,train_iou,val_loss,val_dice,val_iou,epoch_time_sec\n",
                encoding="utf-8",
            )

    best_ckpt_path = outdir / "checkpoints" / "best.pt"
    last_ckpt_path = outdir / "checkpoints" / "last.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, amp=amp, grad_clip=grad_clip
        )
        va_loss, va_dice, va_iou = validate_one_epoch(model, val_loader, criterion, device, amp=amp)

        lr = get_lr(optimizer)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_dice)
            else:
                scheduler.step()

        epoch_time = time.time() - t0

        # Early stopping
        should_stop, is_best = es.step(va_dice)

        # Save last
        save_checkpoint(
            last_ckpt_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val_dice": es.best,
                "config": cfg,
            },
        )

        if is_best:
            save_checkpoint(
                best_ckpt_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_val_dice": es.best,
                    "config": cfg,
                },
            )

        # Print log
        print(
            f"[Epoch {epoch:03d}] lr={lr:.2e} "
            f"train_loss={tr_loss:.4f}, train_dice={tr_dice:.4f}, train_iou={tr_iou:.4f} | "
            f"val_loss={va_loss:.4f}, val_dice={va_dice:.4f}, val_iou={va_iou:.4f} "
            f"(best_val_dice={es.best:.4f}) time={epoch_time:.1f}s"
        )

        # CSV log
        if write_csv:
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{lr:.8f},{tr_loss:.6f},{tr_dice:.6f},{tr_iou:.6f},"
                    f"{va_loss:.6f},{va_dice:.6f},{va_iou:.6f},{epoch_time:.3f}\n"
                )

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch} (best_val_dice={es.best:.4f}).")
            break

    print("Training complete.")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
