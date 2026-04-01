#!/usr/bin/env python3
"""
Render quick visualization artifacts for the Xperience-10M sample dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.data_analysis.xperience_sample_utils import XperienceSampleDataset, resolve_dataset_dir
from scripts.visualization.xperience_sample_visualization_utils import (
    RESAMPLE_LANCZOS,
    build_storyboard,
    render_frame_dashboard,
    save_gif,
)


DEFAULT_OUTPUT_DIR = Path("data_tmp/xperience_sample_visualizations")


def unique_linspace_indices(length: int, count: int) -> list[int]:
    values = np.linspace(0, max(length - 1, 0), num=max(count, 1))
    return sorted({int(round(value)) for value in values})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the Xperience-10M sample dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root that contains annotation.hdf5 and MP4 files. Defaults to $XPERIENCE_SAMPLE_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write storyboard, dashboards, and GIF.",
    )
    parser.add_argument(
        "--main-camera",
        type=str,
        default="stereo_left",
        help="Primary camera used for the storyboard and dashboard main pane.",
    )
    parser.add_argument("--storyboard-frames", type=int, default=6)
    parser.add_argument("--dashboard-frames", type=int, default=3)
    parser.add_argument("--gif-frames", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with XperienceSampleDataset(dataset_dir) as dataset:
        storyboard_indices = unique_linspace_indices(len(dataset), args.storyboard_frames)
        dashboard_indices = unique_linspace_indices(len(dataset), args.dashboard_frames)
        gif_indices = unique_linspace_indices(len(dataset), args.gif_frames)
        secondary_cameras = tuple(stream for stream in ("stereo_right", "fisheye_cam1") if stream in dataset.stream_names)

        storyboard = build_storyboard(dataset, storyboard_indices, main_camera=args.main_camera)
        storyboard_path = output_dir / "storyboard.jpg"
        storyboard.save(storyboard_path, quality=92)

        dashboard_paths: list[str] = []
        gif_frames: list[Image.Image] = []

        for index in dashboard_indices:
            sample = dataset.get_frame(
                index,
                video_streams=(args.main_camera, *secondary_cameras),
                load_video=True,
                load_depth=True,
                load_mocap=True,
                load_imu=True,
                imu_radius=12,
            )
            dashboard = render_frame_dashboard(
                dataset,
                sample,
                main_camera=args.main_camera,
                secondary_cameras=secondary_cameras,
            )
            path = output_dir / f"dashboard_frame_{index:04d}.jpg"
            dashboard.save(path, quality=92)
            dashboard_paths.append(str(path))

        for index in gif_indices:
            sample = dataset.get_frame(
                index,
                video_streams=(args.main_camera, *secondary_cameras),
                load_video=True,
                load_depth=True,
                load_mocap=True,
                load_imu=True,
                imu_radius=12,
            )
            dashboard = render_frame_dashboard(
                dataset,
                sample,
                main_camera=args.main_camera,
                secondary_cameras=secondary_cameras,
            )
            gif_frames.append(dashboard.resize((1000, 612), RESAMPLE_LANCZOS))

    gif_path = output_dir / "sample_animation.gif"
    save_gif(gif_frames, gif_path, duration_ms=180)

    payload = {
        "storyboard_indices": storyboard_indices,
        "dashboard_indices": dashboard_indices,
        "gif_indices": gif_indices,
        "storyboard": str(storyboard_path),
        "dashboards": dashboard_paths,
        "gif": str(gif_path),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
