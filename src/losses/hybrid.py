# src/losses/hybrid.py
# Hybrid Dice + Focal loss for binary segmentation (logits in, {0,1} masks).
#
# Exposes:
# - DiceLoss
# - FocalLoss
# - HybridDiceFocalLoss
#
# Usage:
#   from src.losses.hybrid import HybridDiceFocalLoss
#   loss_fn = HybridDiceFocalLoss(lam=0.5, alpha=0.25, gamma=2.0)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    logits: (N,1,H,W) or (N,H,W)
    targets: same shape, float in {0,1} (or [0,1])
    """
    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = bool(from_logits)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits) if self.from_logits else logits
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.float().contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Binary focal loss (logits version).
    logits: (N,1,H,W) or (N,H,W)
    targets: same shape, float in {0,1}
    alpha: class weighting for positive class
    gamma: focusing parameter
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # BCE with logits per-pixel
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)

        # pt: probability of the true class
        pt = targets * p + (1.0 - targets) * (1.0 - p)

        # alpha balancing
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # focal factor
        focal = alpha_t * (1.0 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class HybridDiceFocalLoss(nn.Module):
    """
    L = lam * DiceLoss + (1 - lam) * FocalLoss
    """
    def __init__(
        self,
        lam: float = 0.5,
        dice_smooth: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0,1].")
        self.lam = float(lam)
        self.dice = DiceLoss(smooth=dice_smooth, from_logits=True)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dl = self.dice(logits, targets)
        fl = self.focal(logits, targets)
        return self.lam * dl + (1.0 - self.lam) * fl
