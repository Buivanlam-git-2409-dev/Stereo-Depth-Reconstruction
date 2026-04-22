from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from src.matching import (
    cosine_similarity_matching,
    pixel_wise_matching,
    window_based_matching,
)
from src.utils import ensure_dir, load_grayscale, normalize_disparity, save_disparity

Method = Literal["pixel", "window", "cosine"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stereo depth reconstruction from rectified image pairs."
    )
    parser.add_argument("--left", required=True, help="Path to left image.")
    parser.add_argument("--right", required=True, help="Path to right image.")
    parser.add_argument(
        "--method",
        default="window",
        choices=["pixel", "window", "cosine"],
        help="Matching method to use.",
    )
    parser.add_argument(
        "--metric",
        default="l1",
        choices=["l1", "l2"],
        help="Distance metric for pixel/window matching.",
    )
    parser.add_argument("--max-disparity", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-prefix", default="disparity")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    left = load_grayscale(args.left)
    right = load_grayscale(args.right)

    if args.method == "pixel":
        disparity = pixel_wise_matching(
            left, right, args.max_disparity, metric=args.metric
        )
        label = f"pixel_{args.metric}"
    elif args.method == "window":
        disparity = window_based_matching(
            left,
            right,
            args.max_disparity,
            metric=args.metric,
            window_size=args.window_size,
        )
        label = f"window_{args.metric}_k{args.window_size}"
    else:
        disparity = cosine_similarity_matching(
            left, right, args.max_disparity, window_size=args.window_size
        )
        label = f"cosine_k{args.window_size}"

    ensure_dir(args.output_dir)
    gray = normalize_disparity(disparity, args.max_disparity)

    prefix = f"{args.output_prefix}_{label}"
    gray_path = str(Path(args.output_dir) / f"{prefix}_gray.png")
    color_path = str(Path(args.output_dir) / f"{prefix}_color.png")
    save_disparity(gray, gray_path, color_path)

    print(f"Saved: {gray_path}")
    print(f"Saved: {color_path}")


if __name__ == "__main__":
    main()
