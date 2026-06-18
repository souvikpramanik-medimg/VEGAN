"""
VEGAN: Vision-language and Edge-enhanced GAN for Microscopic Medical Image Segmentation
Attention U-Net Generator with CBAM and VLM (CLIP) skip connections.

Paper: https://doi.org/10.1109/MECO66322.2025.11049114
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# CBAM: Convolutional Block Attention Module
# ─────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ─────────────────────────────────────────────
# VLM-aware Attention Gate
# ─────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """
    Attention gate fusing decoder features (g), encoder skip features (x),
    and a projected CLIP embedding (vlm) for semantic-aware gating.
    """

    def __init__(self, F_g: int, F_l: int, vlm_dim: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, 1),
            nn.BatchNorm2d(F_l),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l + vlm_dim, F_l, 1),
            nn.BatchNorm2d(F_l),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
        vlm: torch.Tensor,
    ) -> torch.Tensor:
        vlm_spatial = F.interpolate(vlm, size=x.shape[2:], mode="bilinear", align_corners=False)
        x_vlm = torch.cat([x, vlm_spatial], dim=1)
        g1 = self.W_g(g)
        x1 = self.W_x(x_vlm)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)


# ─────────────────────────────────────────────
# Attention U-Net Generator
# ─────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net used as the generator in VEGAN.

    Inputs
    ------
    x            : (B, 2, H, W)  — raw image channel + fused edge map channel
    vlm_features : (B, vlm_feature_size)  — CLIP image embeddings

    Output
    ------
    mask         : (B, 1, H, W) in [0, 1]
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1, vlm_feature_size: int = 512):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.cbam1 = CBAM(64)
        self.enc2 = self._conv_block(64, 128)
        self.cbam2 = CBAM(128)
        self.enc3 = self._conv_block(128, 256)
        self.cbam3 = CBAM(256)
        self.enc4 = self._conv_block(256, 512)
        self.cbam4 = CBAM(512)
        self.pool = nn.MaxPool2d(2)

        # VLM projections (one per skip level)
        self.vlm_proj1 = nn.Sequential(
            nn.Linear(vlm_feature_size, 64),
            nn.Unflatten(1, (64, 1, 1)),
        )
        self.vlm_proj2 = nn.Sequential(
            nn.Linear(vlm_feature_size, 128),
            nn.Unflatten(1, (128, 1, 1)),
        )
        self.vlm_proj3 = nn.Sequential(
            nn.Linear(vlm_feature_size, 256),
            nn.Unflatten(1, (256, 1, 1)),
        )
        self.vlm_proj4 = nn.Sequential(
            nn.Linear(vlm_feature_size, 512),
            nn.Unflatten(1, (512, 1, 1)),
        )

        # Decoder with VLM-aware attention gates
        self.dec1 = self._upconv_block(512, 256)
        self.att1 = AttentionBlock(256, 256, vlm_dim=256)
        self.dec2 = self._upconv_block(512, 128)
        self.att2 = AttentionBlock(128, 128, vlm_dim=128)
        self.dec3 = self._upconv_block(256, 64)
        self.att3 = AttentionBlock(64, 64, vlm_dim=64)

        self.final = nn.Conv2d(128, out_channels, 1)

    # ------------------------------------------------------------------
    def _conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, vlm_features: torch.Tensor) -> torch.Tensor:
        # Project VLM embeddings to each skip level
        vlm1 = self.vlm_proj1(vlm_features)
        vlm2 = self.vlm_proj2(vlm_features)
        vlm3 = self.vlm_proj3(vlm_features)

        # Encoder
        e1 = self.cbam1(self.enc1(x))
        e2 = self.cbam2(self.enc2(self.pool(e1)))
        e3 = self.cbam3(self.enc3(self.pool(e2)))
        e4 = self.cbam4(self.enc4(self.pool(e3)))

        # Decoder
        d1 = self.dec1(e4)
        a1 = self.att1(d1, e3, vlm3)
        d1 = torch.cat([a1, d1], dim=1)

        d2 = self.dec2(d1)
        a2 = self.att2(d2, e2, vlm2)
        d2 = torch.cat([a2, d2], dim=1)

        d3 = self.dec3(d2)
        a3 = self.att3(d3, e1, vlm1)
        d3 = torch.cat([a3, d3], dim=1)

        return torch.sigmoid(self.final(d3))
