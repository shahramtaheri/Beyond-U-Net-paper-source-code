# src/models/attention.py
# Attention gate used to weight skip connection features.

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention gate (Attention U-Net style).
    Inputs:
      x: encoder features (skip)            shape (N, Cx, H, W)
      g: gating signal from decoder node    shape (N, Cg, H, W)
    Output:
      attended x with spatial attention map.
    """
    def __init__(self, x_ch: int, g_ch: int, inter_ch: int | None = None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(1, x_ch // 2)

        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=True)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=True)

        self.bn = nn.BatchNorm2d(inter_ch)
        self.act = nn.ReLU(inplace=True)

        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # assumes x and g are spatially aligned
        theta = self.theta_x(x)
        phi = self.phi_g(g)
        f = self.act(self.bn(theta + phi))
        alpha = self.sigmoid(self.psi(f))  # (N,1,H,W)
        return x * alpha
