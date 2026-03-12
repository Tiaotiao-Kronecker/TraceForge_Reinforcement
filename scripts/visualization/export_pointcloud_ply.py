#!/usr/bin/env python3
"""导出第一帧的 PLY 点云，用于验证外参格式"""
import os
import sys
import numpy as np
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from utils.threed_utils import unproject_by_depth


def save_ply(points, colors, output_path):
    """保存点云为 PLY 格式"""
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def export_frame_pointcloud(episode_dir, frame_idx=0, downsample=2, output_path=None):
    """导出指定帧的点云"""
    from PIL import Image

    episode_dir = Path(episode_dir)
    video_name = episode_dir.name
    main_npz = episode_dir / f"{video_name}.npz"

    data = np.load(main_npz)
    depths = data["depths"].astype(np.float32)
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]

    depth = depths[frame_idx]
    K = intrinsics[frame_idx]
    w2c = extrinsics[frame_idx]
    c2w = np.linalg.inv(w2c)
    H, W = depth.shape

    images_dir = episode_dir / "images"
    img_path = images_dir / f"{video_name}_{frame_idx}.png"

    if img_path.exists():
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = np.array(Image.fromarray(rgb.astype(np.uint8)).resize((W, H))).astype(np.float32)
    else:
        rgb = np.ones((H, W, 3), dtype=np.float32) * 128

    data.close()

    depth_batch = depth[None, None, :, :]
    K_batch = K[None, :, :]
    c2w_batch = c2w[None, :, :]

    xyz = unproject_by_depth(depth_batch, K_batch, c2w_batch)
    xyz = xyz[0].transpose(1, 2, 0)

    pts = xyz[::downsample, ::downsample].reshape(-1, 3)
    colors = rgb[::downsample, ::downsample].reshape(-1, 3)

    valid = (pts[:, 2] > 0) & (pts[:, 2] < 10.0) & np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    colors = colors[valid]

    if output_path is None:
        output_path = episode_dir / f"{video_name}_frame{frame_idx}.ply"

    save_ply(pts, colors, output_path)
    print(f"已保存: {output_path} ({len(pts)} 点)")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    export_frame_pointcloud(args.episode_dir, args.frame_idx, args.downsample, args.output)
