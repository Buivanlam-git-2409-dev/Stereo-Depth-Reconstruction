from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

Metric = Literal["l1", "l2"]


def _shift_right(image: np.ndarray, disparity: int) -> np.ndarray:
    if disparity == 0:
        return image
    shifted = np.zeros_like(image)
    shifted[:, disparity:] = image[:, :-disparity]
    return shifted


def pixel_wise_matching(
    left: np.ndarray,
    right: np.ndarray,
    max_disparity: int,
    metric: Metric = "l1",
) -> np.ndarray:
    """Compute disparity map using pixel-wise matching with vectorized costs."""
    if metric not in {"l1", "l2"}:
        raise ValueError("metric must be 'l1' or 'l2'.")
    if max_disparity <= 0:
        raise ValueError("max_disparity must be positive.")

    best_cost = np.full(left.shape, np.inf, dtype=np.float32)
    best_disp = np.zeros(left.shape, dtype=np.float32)

    for d in range(max_disparity):
        shifted = _shift_right(right, d)
        diff = left - shifted
        cost = np.abs(diff) if metric == "l1" else diff * diff
        if d > 0:
            cost[:, :d] = np.inf
        mask = cost < best_cost
        best_cost[mask] = cost[mask]
        best_disp[mask] = float(d)

    return best_disp


def window_based_matching(
    left: np.ndarray,
    right: np.ndarray,
    max_disparity: int,
    metric: Metric = "l1",
    window_size: int = 5,
) -> np.ndarray:
    """Compute disparity map using window-based cost aggregation."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    if max_disparity <= 0:
        raise ValueError("max_disparity must be positive.")

    best_cost = np.full(left.shape, np.inf, dtype=np.float32)
    best_disp = np.zeros(left.shape, dtype=np.float32)

    for d in range(max_disparity):
        shifted = _shift_right(right, d)
        diff = left - shifted
        cost = np.abs(diff) if metric == "l1" else diff * diff
        if d > 0:
            cost[:, :d] = np.inf
        aggregated = cv2.boxFilter(
            cost,
            ddepth=-1,
            ksize=(window_size, window_size),
            normalize=False,
            borderType=cv2.BORDER_REFLECT,
        )
        mask = aggregated < best_cost
        best_cost[mask] = aggregated[mask]
        best_disp[mask] = float(d)

    return best_disp


def cosine_similarity_matching(
    left: np.ndarray,
    right: np.ndarray,
    max_disparity: int,
    window_size: int = 5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute disparity map using cosine similarity over patches."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    if max_disparity <= 0:
        raise ValueError("max_disparity must be positive.")

    h, w = left.shape
    half = window_size // 2
    valid_h = h - window_size + 1
    valid_w = w - window_size + 1

    left_patches = np.lib.stride_tricks.sliding_window_view(
        left, (window_size, window_size)
    ).reshape(valid_h, valid_w, -1)
    left_norm = np.linalg.norm(left_patches, axis=2, keepdims=True) + eps
    left_unit = left_patches / left_norm

    best_sim = np.full((valid_h, valid_w), -np.inf, dtype=np.float32)
    best_disp = np.zeros((valid_h, valid_w), dtype=np.float32)

    for d in range(max_disparity):
        shifted = _shift_right(right, d)
        right_patches = np.lib.stride_tricks.sliding_window_view(
            shifted, (window_size, window_size)
        ).reshape(valid_h, valid_w, -1)
        right_norm = np.linalg.norm(right_patches, axis=2, keepdims=True) + eps
        right_unit = right_patches / right_norm
        sim = np.sum(left_unit * right_unit, axis=2)

        invalid_cols = max(0, d - half)
        if invalid_cols > 0:
            sim[:, :invalid_cols] = -np.inf

        mask = sim > best_sim
        best_sim[mask] = sim[mask]
        best_disp[mask] = float(d)

    disparity_full = np.zeros((h, w), dtype=np.float32)
    disparity_full[half : h - half, half : w - half] = best_disp
    return disparity_full
