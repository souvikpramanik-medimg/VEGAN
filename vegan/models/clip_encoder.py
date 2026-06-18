"""
VEGAN: CLIP Feature Extractor
Wraps openai/clip-vit-base-patch32 to produce 512-d image embeddings
injected into the Attention U-Net skip connections.

Paper: https://doi.org/10.1109/MECO66322.2025.11049114
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

_MODEL_NAME = "openai/clip-vit-base-patch32"


class CLIPEncoder:
    """
    Lightweight wrapper around HuggingFace CLIP for extracting
    512-dimensional image embeddings.

    Usage
    -----
    encoder = CLIPEncoder(device)
    features = encoder.encode(image_np)   # (1, 512) numpy array
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray
            Grayscale (H, W) or RGB (H, W, 3) uint8 image.

        Returns
        -------
        np.ndarray of shape (1, 512)
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()
