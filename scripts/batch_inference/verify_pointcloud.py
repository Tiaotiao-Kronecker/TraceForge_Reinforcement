#!/usr/bin/env python3
"""验证场景点云重建：检查多帧点云在世界坐标系中是否重叠"""

import numpy as np
import h5py
import os
from pathlib import Path

def unproject_depth_to_world(depth, K, extr, subsample=10):
    """
    将深度图反投影到世界坐标系

    Args:
        depth: (H, W) 深度图，单位米
        K: (3, 3) 内参矩阵
        extr: (4, 4) 外参矩阵（需要确定是c2w还是w2c）
        subsample: 采样间隔（减少点数）

    Returns:
        points_world: (N, 3) 世界坐标系中的3D点
    """
    H, W = depth.shape

    # 创建像素网格（采样）
    y, x = np.meshgrid(
        np.arange(0, H, subsample),
        np.arange(0, W, subsample),
        indexing='ij'
    )

    # 获取对应的深度值
    d = depth[y, x].flatten()
    valid = d > 0

    x = x.flatten()[valid]
    y = y.flatten()[valid]
    d = d[valid]

    # 像素坐标转相机坐标
    xy_homo = np.stack([x, y, np.ones_like(x)], axis=0)  # (3, N)
    K_inv = np.linalg.inv(K)
    cam_coords = K_inv @ xy_homo  # (3, N)
    cam_coords = cam_coords * d[None, :]  # (3, N)

    # 相机坐标转世界坐标
    # 尝试两种方式：
    # 方式1：假设extr是w2c，取逆得到c2w
    c2w_1 = np.linalg.inv(extr)
    world_coords_1 = c2w_1[:3, :3] @ cam_coords + c2w_1[:3, 3:4]

    # 方式2：假设extr已经是c2w
    world_coords_2 = extr[:3, :3] @ cam_coords + extr[:3, 3:4]

    return world_coords_1.T, world_coords_2.T  # (N, 3)


def main():
    # 数据路径
    base_path = Path("/home/zoyo/projects/droid_preprocess_pipeline/droid_raw/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s")
    h5_path = base_path / "trajectory_valid.h5"
    depth_dir = base_path / "depth/hand_camera/depth"

    camera_name = "hand_camera"

    # 读取外参和内参
    with h5py.File(h5_path, 'r') as f:
        intrs = f[f'observation/camera/intrinsics/{camera_name}_left'][:]
        extrs = f[f'observation/camera/extrinsics/{camera_name}_left'][:]

    # 选择几帧进行验证（0, 50, 100, 150, 200）
    test_frames = [0, 50, 100, 150, 200]

    print("=== 场景点云重建验证 ===\n")

    for method_name, method_idx in [("方式1: extr是w2c，取逆得c2w", 0),
                                      ("方式2: extr已经是c2w", 1)]:
        print(f"\n{method_name}")
        print("-" * 60)

        all_points = []

        for frame_idx in test_frames:
            # 读取深度
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
            depth_path = os.path.join(depth_dir, depth_files[frame_idx])
            depth = np.load(depth_path)

            # 反投影
            K = intrs[frame_idx]
            extr = extrs[frame_idx]

            points_1, points_2 = unproject_depth_to_world(depth, K, extr, subsample=20)
            points = points_1 if method_idx == 0 else points_2

            all_points.append(points)

            # 统计每帧点云范围
            print(f"  帧{frame_idx:3d}: X[{points[:, 0].min():7.3f}, {points[:, 0].max():7.3f}] "
                  f"Y[{points[:, 1].min():7.3f}, {points[:, 1].max():7.3f}] "
                  f"Z[{points[:, 2].min():7.3f}, {points[:, 2].max():7.3f}] "
                  f"({len(points)} 点)")

        # 合并所有帧的点云
        all_points = np.concatenate(all_points, axis=0)

        print(f"\n  合并后: X[{all_points[:, 0].min():7.3f}, {all_points[:, 0].max():7.3f}] "
              f"Y[{all_points[:, 1].min():7.3f}, {all_points[:, 1].max():7.3f}] "
              f"Z[{all_points[:, 2].min():7.3f}, {all_points[:, 2].max():7.3f}]")

        # 计算范围
        ranges = all_points.max(axis=0) - all_points.min(axis=0)
        print(f"  范围: X={ranges[0]:.3f}m, Y={ranges[1]:.3f}m, Z={ranges[2]:.3f}m")

        # 判断是否合理
        # 对于室内场景，合理的范围应该在几米以内
        if np.all(ranges < 10):  # 所有维度范围 < 10米
            print(f"  ✅ 范围合理（室内场景）")
        else:
            print(f"  ❌ 范围过大（可能外参使用错误）")


if __name__ == "__main__":
    main()
