# src/models/residual_unetpp_attention.py
# U-Net++ (nested skip connections) using ResidualBlock at each node + AttentionGate on skip features.
#
# Exposes: ResidualAttentionUNetPP(in_ch=1, out_ch=1, filters=(64,128,256,512,1024))

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import ResidualBlock, Up
from src.models.attention import AttentionGate


class ResidualAttentionUNetPP(nn.Module):
    """
    Nested U-Net++ (depth=4) with residual blocks and attention gates.
    Notation: X_{i,j} where i is depth level, j is stage in nested skip pathway.
    Final prediction uses X_{0,4}.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, filters=(64, 128, 256, 512, 1024)):
        super().__init__()
        f0, f1, f2, f3, f4 = filters

        self.pool = nn.MaxPool2d(2, 2)
        self.up = Up()

        # Encoder: X_{i,0}
        self.x00 = ResidualBlock(in_ch, f0)
        self.x10 = ResidualBlock(f0, f1)
        self.x20 = ResidualBlock(f1, f2)
        self.x30 = ResidualBlock(f2, f3)
        self.x40 = ResidualBlock(f3, f4)

        # Attention gates for first-level skip fusion (j=1)
        self.ag3_1 = AttentionGate(x_ch=f3, g_ch=f4)  # x30 gated by up(x40)
        self.ag2_1 = AttentionGate(x_ch=f2, g_ch=f3)  # x20 gated by up(x30)
        self.ag1_1 = AttentionGate(x_ch=f1, g_ch=f2)  # x10 gated by up(x20)
        self.ag0_1 = AttentionGate(x_ch=f0, g_ch=f1)  # x00 gated by up(x10)

        # Attention gates for deeper nested paths (use gating = up(X_{i+1,j-1}))
        self.ag2_2 = AttentionGate(x_ch=f2, g_ch=f3)  # x21 gated by up(x31)
        self.ag1_2 = AttentionGate(x_ch=f1, g_ch=f2)  # x11 gated by up(x21)
        self.ag0_2 = AttentionGate(x_ch=f0, g_ch=f1)  # x01 gated by up(x11)

        self.ag1_3 = AttentionGate(x_ch=f1, g_ch=f2)  # x12 gated by up(x22)
        self.ag0_3 = AttentionGate(x_ch=f0, g_ch=f1)  # x02 gated by up(x12)

        self.ag0_4 = AttentionGate(x_ch=f0, g_ch=f1)  # x03 gated by up(x13)

        # Decoder nested nodes: X_{i,j}, j>0
        # j = 1
        self.x31 = ResidualBlock(f3 + f4, f3)
        self.x21 = ResidualBlock(f2 + f3, f2)
        self.x11 = ResidualBlock(f1 + f2, f1)
        self.x01 = ResidualBlock(f0 + f1, f0)

        # j = 2: concat(X_{i,0}, X_{i,1}, up(X_{i+1,1}))
        self.x22 = ResidualBlock(f2 + f2 + f3, f2)
        self.x12 = ResidualBlock(f1 + f1 + f2, f1)
        self.x02 = ResidualBlock(f0 + f0 + f1, f0)

        # j = 3: concat(X_{i,0}, X_{i,1}, X_{i,2}, up(X_{i+1,2}))
        self.x13 = ResidualBlock(f1 + f1 + f1 + f2, f1)
        self.x03 = ResidualBlock(f0 + f0 + f0 + f1, f0)

        # j = 4: concat(X_{0,0},X_{0,1},X_{0,2},X_{0,3}, up(X_{1,3}))
        self.x04 = ResidualBlock(f0 + f0 + f0 + f0 + f1, f0)

        # Output head
        self.out_conv = nn.Conv2d(f0, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        # j=1
        g31 = self.up(x40)
        x30_att = self.ag3_1(x30, g31)
        x31 = self.x31(torch.cat([x30_att, g31], dim=1))

        g21 = self.up(x30)
        x20_att = self.ag2_1(x20, g21)
        x21 = self.x21(torch.cat([x20_att, g21], dim=1))

        g11 = self.up(x20)
        x10_att = self.ag1_1(x10, g11)
        x11 = self.x11(torch.cat([x10_att, g11], dim=1))

        g01 = self.up(x10)
        x00_att = self.ag0_1(x00, g01)
        x01 = self.x01(torch.cat([x00_att, g01], dim=1))

        # j=2
        g22 = self.up(x31)
        # gate x21 using up(x31) as gating
        x21_att = self.ag2_2(x21, g22)
        x22 = self.x22(torch.cat([x20, x21_att, g22], dim=1))

        g12 = self.up(x21)
        x11_att = self.ag1_2(x11, g12)
        x12 = self.x12(torch.cat([x10, x11_att, g12], dim=1))

        g02 = self.up(x11)
        x01_att = self.ag0_2(x01, g02)
        x02 = self.x02(torch.cat([x00, x01_att, g02], dim=1))

        # j=3
        g13 = self.up(x22)
        x12_att = self.ag1_3(x12, g13)
        x13 = self.x13(torch.cat([x10, x11, x12_att, g13], dim=1))

        g03 = self.up(x12)
        x02_att = self.ag0_3(x02, g03)
        x03 = self.x03(torch.cat([x00, x01, x02_att, g03], dim=1))

        # j=4
        g04 = self.up(x13)
        x03_att = self.ag0_4(x03, g04)
        x04 = self.x04(torch.cat([x00, x01, x02, x03_att, g04], dim=1))

        return self.out_conv(x04)
