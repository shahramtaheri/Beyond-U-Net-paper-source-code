import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Two 3x3 convs + BN + ReLU with identity/projection skip.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, 3, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class AttentionGate(nn.Module):
    """
    Attention U-Net style gate:
      inputs: x (encoder features), g (decoder gating signal)
      output: attended x
    """
    def __init__(self, x_ch, g_ch, inter_ch=None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(1, x_ch // 2)

        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=True)
        self.phi_g   = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=True)
        self.psi     = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)

        self.bn = nn.BatchNorm2d(inter_ch)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x: encoder feature at some resolution
        # g: decoder feature at same resolution
        # shapes should match spatially
        theta = self.theta_x(x)
        phi = self.phi_g(g)
        f = self.act(self.bn(theta + phi))
        alpha = self.sigmoid(self.psi(f))
        return x * alpha


class Up(nn.Module):
    """
    Bilinear upsampling + 1x1 conv optional (keeps channels stable if desired).
    """
    def __init__(self, scale=2, mode="bilinear", align_corners=False):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners)


# ----------------------------
# U-Net++ with Residual blocks + Attention gates
# ----------------------------

class ResidualAttentionUNetPP(nn.Module):
    """
    U-Net++ (nested skip paths) where each node uses ResidualBlock.
    Attention gates are applied to encoder skip features before concatenation.

    Depth: 4 downsamples (0..4 levels), typical filters: [64,128,256,512,1024]
    """
    def __init__(self, in_ch=1, out_ch=1, filters=(64, 128, 256, 512, 1024)):
        super().__init__()
        f0, f1, f2, f3, f4 = filters

        self.pool = nn.MaxPool2d(2, 2)
        self.up = Up()

        # Encoder nodes X_{i,0}
        self.x00 = ResidualBlock(in_ch, f0)
        self.x10 = ResidualBlock(f0, f1)
        self.x20 = ResidualBlock(f1, f2)
        self.x30 = ResidualBlock(f2, f3)
        self.x40 = ResidualBlock(f3, f4)

        # Attention gates (skip from encoder level i to decoder node at same resolution)
        # g_ch depends on the decoder feature used as gating at that resolution.
        # We'll define gates for each resolution where we fuse encoder into a decoder node.
        self.ag0_1 = AttentionGate(x_ch=f0, g_ch=f1)   # for x00 with g = up(x10)
        self.ag1_1 = AttentionGate(x_ch=f1, g_ch=f2)   # for x10 with g = up(x20)
        self.ag2_1 = AttentionGate(x_ch=f2, g_ch=f3)   # for x20 with g = up(x30)
        self.ag3_1 = AttentionGate(x_ch=f3, g_ch=f4)   # for x30 with g = up(x40)

        # For deeper nested nodes, gating channels change (g_ch = channels of upsampled node used)
        self.ag0_2 = AttentionGate(x_ch=f0, g_ch=f1)   # g = up(x11) has f1 channels
        self.ag1_2 = AttentionGate(x_ch=f1, g_ch=f2)   # g = up(x21) has f2 channels
        self.ag2_2 = AttentionGate(x_ch=f2, g_ch=f3)   # g = up(x31) has f3 channels

        self.ag0_3 = AttentionGate(x_ch=f0, g_ch=f1)   # g = up(x12) has f1
        self.ag1_3 = AttentionGate(x_ch=f1, g_ch=f2)   # g = up(x22) has f2

        self.ag0_4 = AttentionGate(x_ch=f0, g_ch=f1)   # g = up(x13) has f1

        # Decoder nested nodes X_{i,j} (j>0)
        # Node formula: X_{i,j} = ResidualBlock( concat( X_{i,0..j-1}, up(X_{i+1,j-1}) ) )
        # Channels: sum(previous at same i) + channels of upsampled from i+1
        self.x01 = ResidualBlock(f0 + f1, f0)
        self.x11 = ResidualBlock(f1 + f2, f1)
        self.x21 = ResidualBlock(f2 + f3, f2)
        self.x31 = ResidualBlock(f3 + f4, f3)

        self.x02 = ResidualBlock(f0 + f0 + f1, f0)
        self.x12 = ResidualBlock(f1 + f1 + f2, f1)
        self.x22 = ResidualBlock(f2 + f2 + f3, f2)

        self.x03 = ResidualBlock(f0 + f0 + f0 + f1, f0)
        self.x13 = ResidualBlock(f1 + f1 + f1 + f2, f1)

        self.x04 = ResidualBlock(f0 + f0 + f0 + f0 + f1, f0)

        # Final output head
        self.out_conv = nn.Conv2d(f0, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        # Level 3 -> 1
        g31 = self.up(x40)
        x30_att = self.ag3_1(x30, g31)
        x31 = self.x31(torch.cat([x30_att, g31], dim=1))

        # Level 2 -> 1
        g21 = self.up(x30)
        x20_att = self.ag2_1(x20, g21)
        x21 = self.x21(torch.cat([x20_att, g21], dim=1))

        # Level 1 -> 1
        g11 = self.up(x20)
        x10_att = self.ag1_1(x10, g11)
        x11 = self.x11(torch.cat([x10_att, g11], dim=1))

        # Level 0 -> 1
        g01 = self.up(x10)
        x00_att = self.ag0_1(x00, g01)
        x01 = self.x01(torch.cat([x00_att, g01], dim=1))

        # j=2 nodes
        g22 = self.up(x31)
        x21_att = self.ag2_2(x21, g22)
        x22 = self.x22(torch.cat([x20, x21_att, g22], dim=1))

        g12 = self.up(x21)
        x11_att = self.ag1_2(x11, g12)
        x12 = self.x12(torch.cat([x10, x11_att, g12], dim=1))

        g02 = self.up(x11)
        x01_att = self.ag0_2(x01, g02)
        x02 = self.x02(torch.cat([x00, x01_att, g02], dim=1))

        # j=3 nodes
        g13 = self.up(x22)
        x12_att = self.ag1_3(x12, g13)
        x13 = self.x13(torch.cat([x10, x11, x12_att, g13], dim=1))

        g03 = self.up(x12)
        x02_att = self.ag0_3(x02, g03)
        x03 = self.x03(torch.cat([x00, x01, x02_att, g03], dim=1))

        # j=4 node
        g04 = self.up(x13)
        x03_att = self.ag0_4(x03, g04)
        x04 = self.x04(torch.cat([x00, x01, x02, x03_att, g04], dim=1))

        logits = self.out_conv(x04)
        return logits


# ----------------------------
# Losses: Dice + Focal (binary)
# ----------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N,1,H,W), targets: (N,1,H,W) in {0,1}
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Binary focal loss
        # targets in {0,1}
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HybridDiceFocalLoss(nn.Module):
    def __init__(self, lam=0.5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.lam = lam
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits, targets):
        return self.lam * self.dice(logits, targets) + (1.0 - self.lam) * self.focal(logits, targets)


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResidualAttentionUNetPP(in_ch=1, out_ch=1).to(device)
    criterion = HybridDiceFocalLoss(lam=0.5, alpha=0.25, gamma=2.0)

    # dummy batch: N=2, 1x512x512
    x = torch.randn(2, 1, 512, 512, device=device)
    y = (torch.rand(2, 1, 512, 512, device=device) > 0.95).float()  # sparse positives

    logits = model(x)
    loss = criterion(logits, y)

    print("logits:", logits.shape, "loss:", float(loss))
