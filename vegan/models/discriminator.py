"""
VEGAN: PatchGAN Discriminator
Based on the Pix2Pix framework (Isola et al., CVPR 2017).

Paper: https://doi.org/10.1109/MECO66322.2025.11049114
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN discriminator that classifies overlapping image patches as
    real or fake, encouraging fine-grained structural coherence.

    Input
    -----
    x : (B, 2, H, W)  — concatenation of [image_channel, mask]

    Output
    ------
    patch_logits : (B, 1, H', W')  — per-patch real/fake scores
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1 — no BatchNorm on first layer (following Pix2Pix convention)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output — 1-channel patch map
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
