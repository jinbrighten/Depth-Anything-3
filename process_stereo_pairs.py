#!/usr/bin/env python3
"""
Multi-view stereo pair processing script for Depth Anything 3.

Processes paired images from two camera streams (stream_1201-1 and stream_1201-2)
by matching timestamps in filenames.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


def get_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename like '44335766666666_undistorted.jpg'."""
    return filename.split("_")[0]


def find_stereo_pairs(stream1_dir: Path, stream2_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching image pairs between two stream directories based on timestamps."""
    # Get all images from both streams
    stream1_images = {
        get_timestamp_from_filename(f.name): f
        for f in stream1_dir.glob("*_undistorted.jpg")
    }
    stream2_images = {
        get_timestamp_from_filename(f.name): f
        for f in stream2_dir.glob("*_undistorted.jpg")
    }

    # Find common timestamps
    common_timestamps = sorted(set(stream1_images.keys()) & set(stream2_images.keys()))

    pairs = [
        (stream1_images[ts], stream2_images[ts])
        for ts in common_timestamps
    ]

    return pairs


def process_stereo_pairs(
    data_dir: Path,
    output_dir: Path,
    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    export_format: str = "npz",
    device: str = "cuda",
) -> None:
    """Process all stereo pairs from the data directory."""
    stream1_dir = data_dir / "stream_1201-1"
    stream2_dir = data_dir / "stream_1201-2"

    if not stream1_dir.exists():
        raise FileNotFoundError(f"Stream 1 directory not found: {stream1_dir}")
    if not stream2_dir.exists():
        raise FileNotFoundError(f"Stream 2 directory not found: {stream2_dir}")

    # Find all pairs
    pairs = find_stereo_pairs(stream1_dir, stream2_dir)
    print(f"Found {len(pairs)} stereo pairs")

    if not pairs:
        print("No matching pairs found!")
        return

    # Load model
    print(f"Loading model: {model_dir}")
    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device=device)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each pair
    for left_path, right_path in tqdm(pairs, desc="Processing pairs"):
        timestamp = get_timestamp_from_filename(left_path.name)
        pair_output_dir = output_dir / f"pair_{timestamp}"

        # Process the pair together (both views in one forward pass)
        prediction = model.inference(
            [str(left_path), str(right_path)],
            export_dir=str(pair_output_dir),
            export_format=export_format,
        )

        # prediction.depth has shape [2, H, W] for the pair
        # prediction.extrinsics has shape [2, 3, 4] for camera poses
        # prediction.intrinsics has shape [2, 3, 3] for camera intrinsics


def main():
    parser = argparse.ArgumentParser(
        description="Process stereo image pairs with Depth Anything 3"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing stream_1201-1 and stream_1201-2 folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data_dir/depth_output)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Model to use",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["npz", "glb", "ply", "gs_video"],
        default="npz",
        help="Output format",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or (args.data_dir / "depth_output")

    process_stereo_pairs(
        data_dir=args.data_dir,
        output_dir=output_dir,
        model_dir=args.model_dir,
        export_format=args.export_format,
        device=args.device,
    )


if __name__ == "__main__":
    main()
