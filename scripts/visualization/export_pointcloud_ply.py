#!/usr/bin/env python3
"""导出指定帧的 PLY 点云，兼容 v2/legacy TraceForge 输出。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import SceneReader, build_pointcloud_from_frame


def save_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            x, y, z = point
            r, g, b = color
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def export_frame_pointcloud(
    episode_dir: str | Path,
    frame_idx: int = 0,
    downsample: int = 2,
    output_path: str | Path | None = None,
) -> Path:
    episode_dir = Path(episode_dir).resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    with SceneReader(episode_dir) as scene_reader:
        intrinsics, extrinsics = scene_reader.get_camera_arrays()
        if frame_idx >= len(intrinsics) or frame_idx >= len(extrinsics):
            raise IndexError(
                f"frame_idx={frame_idx} exceeds available camera frames ({len(intrinsics)})"
            )
        depth = scene_reader.get_depth_frame(frame_idx)
        rgb = scene_reader.get_rgb_frame(frame_idx)
        points, colors = build_pointcloud_from_frame(
            depth=depth,
            rgb=rgb,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics[frame_idx],
            downsample=max(1, downsample),
        )

    colors_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    if output_path is None:
        output_path = episode_dir / f"{episode_dir.name}_frame{frame_idx}.ply"
    output_path = Path(output_path)
    save_ply(points, colors_u8, output_path)
    print(f"已保存: {output_path} ({len(points)} 点)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    export_frame_pointcloud(
        args.episode_dir,
        frame_idx=args.frame_idx,
        downsample=args.downsample,
        output_path=args.output,
    )
