"""
VEGAN Inference Script
======================
Run a trained VEGAN generator on a single image or a directory of images.

Usage
-----
python vegan/inference.py \\
    --checkpoint checkpoints/best_generator.pth \\
    --input      path/to/image.png \\
    --output     results/

Paper: https://doi.org/10.1109/MECO66322.2025.11049114
"""

import argparse
import os

import cv2
import numpy as np
import torch

from vegan.models.attention_unet import AttentionUNet
from vegan.models.clip_encoder import CLIPEncoder
from vegan.utils.edge_maps import get_edge_map


def load_generator(checkpoint: str, device: torch.device) -> AttentionUNet:
    model = AttentionUNet(vlm_feature_size=512).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image_path: str, clip_enc: CLIPEncoder, device: torch.device):
    """Load image → edge map + CLIP features → tensors ready for inference."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"Cannot read image: {image_path}"
    image = cv2.resize(image, (256, 256))

    edges = get_edge_map(image)
    combined = np.stack([image, (edges * 255).astype(np.uint8)], axis=0).astype(np.float32)
    img_tensor = torch.from_numpy(combined).unsqueeze(0).to(device)   # (1, 2, 256, 256)

    image_rgb = np.stack([image] * 3, axis=-1)
    vlm_np = clip_enc.encode(image_rgb)
    vlm_tensor = torch.tensor(vlm_np, dtype=torch.float32).to(device)  # (1, 512)

    return img_tensor, vlm_tensor, image


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_enc = CLIPEncoder(device)
    generator = load_generator(args.checkpoint, device)
    os.makedirs(args.output, exist_ok=True)

    # Collect input paths
    if os.path.isdir(args.input):
        paths = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith((".png", ".jpg", ".tif", ".bmp"))
        ])
    else:
        paths = [args.input]

    for path in paths:
        img_tensor, vlm_tensor, orig = preprocess(path, clip_enc, device)
        with torch.no_grad():
            pred = generator(img_tensor, vlm_tensor)         # (1, 1, 256, 256)

        mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        out_name = os.path.splitext(os.path.basename(path))[0] + "_mask.png"
        out_path = os.path.join(args.output, out_name)
        cv2.imwrite(out_path, mask)
        print(f"  Saved: {out_path}")

    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VEGAN inference on medical images.")
    p.add_argument("--checkpoint", required=True, help="Path to best_generator.pth")
    p.add_argument("--input",      required=True, help="Image file or directory")
    p.add_argument("--output",     default="results/", help="Output directory for masks")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
