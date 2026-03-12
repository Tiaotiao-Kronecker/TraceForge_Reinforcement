#!/usr/bin/env python3
"""从 depth 和 images 目录直接导出 PLY 点云"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from utils.threed_utils import unproject_by_depth


def save_ply(points, colors, output_path):
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


def export_from_depth_images(episode_dir, video_name, frame_idx, h5_path, camera_name, extr_mode, downsample=2):
    """从 depth/images 目录和 H5 外参导出点云"""
    episode_dir = Path(episode_dir)

    # 读取深度图
    depth_raw_path = episode_dir / "depth" / f"{video_name}_{frame_idx}_raw.npz"
    depth_data = np.load(depth_raw_path)
    depth = depth_data["depth"].astype(np.float32)
    depth_data.close()

    # 读取 RGB
    img_path = episode_dir / "images" / f"{video_name}_{frame_idx}.png"
    rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)

    H, W = depth.shape
    if rgb.shape[0] != H or rgb.shape[1] != W:
        rgb = np.array(Image.fromarray(rgb.astype(np.uint8)).resize((W, H))).astype(np.float32)

    # 从 H5 读取外参
    import h5py
    with h5py.File(h5_path, 'r') as f:
        intr_key = f"observation/camera/intrinsics/{camera_name}_left"
        extr_key = f"observation/camera/extrinsics/{camera_name}_left"
        if intr_key not in f:
            intr_key = f"observation/camera/intrinsics/{camera_name}"
            extr_key = f"observation/camera/extrinsics/{camera_name}"

        K = f[intr_key][frame_idx].astype(np.float32)
        extrs_raw = f[extr_key][frame_idx].astype(np.float32)

    # 处理外参模式
    if extr_mode == "w2c":
        w2c = extrs_raw
    else:  # c2w
        w2c = np.linalg.inv(extrs_raw)

    c2w = np.linalg.inv(w2c)

    # 反投影
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

    return pts, colors


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--camera_name", type=str, required=True)
    parser.add_argument("--extr_mode", type=str, required=True, choices=["w2c", "c2w"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()

    pts, colors = export_from_depth_images(
        args.episode_dir, args.video_name, args.frame_idx,
        args.h5_path, args.camera_name, args.extr_mode, args.downsample
    )
    save_ply(pts, colors, args.output)
    print(f"已保存: {args.output} ({len(pts)} 点)")
