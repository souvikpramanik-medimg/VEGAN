"""
Loss functions used in VEGAN training.

Generator loss = Dice Loss + Adversarial Loss
Discriminator loss = BCE Loss
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation: maximizes overlap between predicted and ground truth masks.

    Loss = 1 - (2 * |P ∩ T|) / (|P| + |T|)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth
        )
        return 1.0 - dice.mean()


class BCELoss(nn.Module):
    """Standard Binary Cross-Entropy loss."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(pred, target)
