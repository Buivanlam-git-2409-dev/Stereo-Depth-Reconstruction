from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_grayscale(image_path: str) -> np.ndarray:
    """Load an image as grayscale float32."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image.astype(np.float32)


def normalize_disparity(disparity: np.ndarray, max_disparity: int) -> np.ndarray:
    """Normalize disparity values to 0-255 for visualization."""
    if max_disparity <= 0:
        raise ValueError("max_disparity must be positive.")
    max_scale = max(1, max_disparity - 1)
    scaled = np.clip(disparity, 0, max_scale) * (255.0 / max_scale)
    return scaled.astype(np.uint8)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_disparity(
    disparity_gray: np.ndarray,
    gray_path: str,
    color_path: str,
) -> Tuple[str, str]:
    """Save grayscale and color-mapped disparity images."""
    color_map = cv2.applyColorMap(disparity_gray, cv2.COLORMAP_JET)
    cv2.imwrite(gray_path, disparity_gray)
    cv2.imwrite(color_path, color_map)
    return gray_path, color_path
