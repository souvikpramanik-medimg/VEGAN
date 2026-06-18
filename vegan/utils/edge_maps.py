"""
Combined Edge Map computation using Sobel, Canny, and Laplacian operators.

As described in Section III-B of the VEGAN paper:
    Pramanik et al., MECO 2025, DOI: 10.1109/MECO66322.2025.11049114
"""

import cv2
import numpy as np


def sobel_edge(image: np.ndarray) -> np.ndarray:
    """Compute Sobel edge map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    return sobel


def canny_edge(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Compute Canny edge map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_uint8 = np.uint8(gray)
    return cv2.Canny(gray_uint8, low, high).astype(np.float64)


def laplacian_edge(image: np.ndarray) -> np.ndarray:
    """Compute Laplacian of Gaussian edge map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.abs(lap)


def compute_combined_edge_map(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """
    Compute a fused edge map by averaging Sobel, Canny, and Laplacian edge responses.
    All individual maps are min-max normalized before averaging.

    Args:
        image: Input image (H x W x C) or (H x W), uint8.
        canny_low: Lower threshold for Canny hysteresis.
        canny_high: Upper threshold for Canny hysteresis.

    Returns:
        Combined edge map in [0, 1], shape (H, W).
    """
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mn) / (mx - mn)).astype(np.float32)

    sobel = normalize(sobel_edge(image))
    canny = normalize(canny_edge(image, canny_low, canny_high))
    laplacian = normalize(laplacian_edge(image))

    combined = (sobel + canny + laplacian) / 3.0
    return combined
