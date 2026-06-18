"""
VEGAN Training Script
=====================
Trains the VEGAN generator (Attention U-Net + CLIP) and PatchGAN discriminator
on a binary medical image segmentation dataset.

Usage
-----
python vegan/train.py \\
    --dataset monuseg \\
    --image_dir data/MoNuSeg/train/images \\
    --mask_dir  data/MoNuSeg/train/masks \\
    --val_image_dir data/MoNuSeg/test/images \\
    --val_mask_dir  data/MoNuSeg/test/masks \\
    --epochs 200 --batch_size 2 --lr 1e-3

Paper: https://doi.org/10.1109/MECO66322.2025.11049114
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from vegan.data.dataset import MedicalSegDataset, get_train_transform, get_val_transform, load_paths
from vegan.models.attention_unet import AttentionUNet
from vegan.models.clip_encoder import CLIPEncoder
from vegan.models.discriminator import Discriminator
from vegan.utils.losses import dice_loss
from vegan.utils.metrics import dice_score, iou_score


# ─── Loss helpers ─────────────────────────────────────────────────────

bce = nn.BCEWithLogitsLoss()


def generator_loss(
    fake_masks: torch.Tensor,
    real_masks: torch.Tensor,
    discriminator: Discriminator,
    images: torch.Tensor,
    lambda_dice: float = 1.0,
) -> torch.Tensor:
    fake_input = torch.cat([images[:, :1], fake_masks], dim=1)
    g_pred = discriminator(fake_input)
    adv_loss = bce(g_pred, torch.ones_like(g_pred))
    seg_loss = dice_loss(fake_masks, real_masks)
    return adv_loss + lambda_dice * seg_loss


def discriminator_loss(
    fake_masks: torch.Tensor,
    real_masks: torch.Tensor,
    discriminator: Discriminator,
    images: torch.Tensor,
) -> torch.Tensor:
    real_input = torch.cat([images[:, :1], real_masks], dim=1)
    fake_input = torch.cat([images[:, :1], fake_masks.detach()], dim=1)
    real_loss = bce(discriminator(real_input), torch.ones_like(discriminator(real_input)))
    fake_loss = bce(discriminator(fake_input), torch.zeros_like(discriminator(fake_input)))
    return (real_loss + fake_loss) / 2


# ─── Training loop ────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_enc = CLIPEncoder(device)

    train_imgs, train_masks = load_paths(args.image_dir, args.mask_dir)
    val_imgs, val_masks = load_paths(args.val_image_dir, args.val_mask_dir)

    train_ds = MedicalSegDataset(train_imgs, train_masks, clip_enc, get_train_transform())
    val_ds = MedicalSegDataset(val_imgs, val_masks, clip_enc, get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    generator = AttentionUNet(vlm_feature_size=512).to(device)
    discriminator = Discriminator().to(device)

    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        train_dice = train_iou = 0.0

        for images, masks, vlm_features in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images = images.to(device)
            masks = masks.to(device)
            vlm_features = vlm_features.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            # ── Discriminator step ──
            opt_d.zero_grad()
            with torch.no_grad():
                fake_masks = generator(images, vlm_features)
            d_loss = discriminator_loss(fake_masks, masks, discriminator, images)
            d_loss.backward()
            opt_d.step()

            # ── Generator step ──
            opt_g.zero_grad()
            fake_masks = generator(images, vlm_features)
            g_loss = generator_loss(fake_masks, masks, discriminator, images, args.lambda_dice)
            g_loss.backward()
            opt_g.step()

            train_dice += dice_score(fake_masks.detach(), masks)
            train_iou += iou_score(fake_masks.detach(), masks)

        # ── Validation ──
        generator.eval()
        val_dice = val_iou = 0.0
        with torch.no_grad():
            for images, masks, vlm_features in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  "):
                images = images.to(device)
                masks = masks.to(device)
                vlm_features = vlm_features.to(device)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                preds = generator(images, vlm_features)
                val_dice += dice_score(preds, masks)
                val_iou += iou_score(preds, masks)

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Dice {train_dice/len(train_loader):.4f} | "
            f"Val Dice {val_dice:.4f} | Val IoU {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(generator.state_dict(), "checkpoints/best_generator.pth")
            print(f"  ✓ Saved best model (val Dice = {best_val_dice:.4f})")

    print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VEGAN on a medical image segmentation dataset.")
    p.add_argument("--dataset",       default="monuseg", choices=["monuseg", "cns", "ph2"])
    p.add_argument("--image_dir",     required=True)
    p.add_argument("--mask_dir",      required=True)
    p.add_argument("--val_image_dir", required=True)
    p.add_argument("--val_mask_dir",  required=True)
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--lambda_dice",   type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
