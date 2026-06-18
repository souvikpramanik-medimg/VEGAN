"""
Evaluation metrics: Dice Score and IoU (Jaccard Index).
"""

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient between predicted and target binary masks.

    Args:
        pred: Predicted mask (B, 1, H, W), values in [0, 1].
        target: Ground truth mask (B, 1, H, W), binary.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Mean Dice score across the batch.
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute Intersection over Union (Jaccard Index).

    Args:
        pred: Predicted mask (B, 1, H, W), values in [0, 1].
        target: Ground truth mask (B, 1, H, W), binary.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Mean IoU score across the batch.
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
