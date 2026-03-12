#!/usr/bin/env python3
"""
TraceForge 推理结果的 3D Keypoint 动画可视化

对每个 case，按查询帧分别加载轨迹，从第 0 帧到第 T 帧播放 keypoint 的 3D 动画。
类似 SpaTrackerV2 的 3D 可视化，但仅对 grid 采样的 keypoint 做动画（非逐像素）。

用法:
    python visualize_3d_keypoint_animation.py \
        --episode_dir output_bridge_depth_grid80/00000/images0 \
        [--query_frame 0] \           # 指定查询帧，默认第一个可用
        [--keypoint_stride 10] \      # 每 N 个显示 1 个；减小可显示更多（如 2->3200 点，1->6400 点）
        [--dense_pointcloud] \        # 首帧查询时：显示完整密集点云（类似 SpaTrackerV2）
        [--dense_downsample 4] \      # 密集点云下采样因子
        [--normalize_camera] \        # 变换到首帧相机坐标系（类似 SpaTrackerV2）
        [--port 8080]

    不同查询帧需分别指定 --query_frame 0/5/10/... 后重新运行。
"""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path

# 添加项目根目录
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

import viser
from matplotlib import colormaps
from loguru import logger

from utils.threed_utils import unproject_by_depth


def load_sample_npz(npz_path):
    """加载单个 sample NPZ"""
    data = np.load(npz_path)
    traj = data["traj"]  # (N, T, 3)

    # 兼容没有 valid_steps 的情况（禁用retarget后）
    if "valid_steps" in data:
        valid_steps = data["valid_steps"]  # (T,)
    else:
        # 根据inf值自动生成valid_steps
        # 如果某个时间步的所有轨迹都不是inf，则认为该步有效
        valid_steps = ~np.all(np.isinf(traj), axis=(0, 2))  # (T,)

    keypoints = data["keypoints"]  # (N, 2)
    frame_index = int(data["frame_index"][0])
    data.close()
    return traj, valid_steps, keypoints, frame_index


def traj_pixel_depth_to_world(traj_uvz, K, w2c):
    """
    将 sample NPZ 的 traj 从 (u_pixel, v_pixel, z_depth) 转为世界坐标。
    infer 中 project_tracks_3d_to_3d 输出 (fx*x/z+cx, fy*y/z+cy, z)，需反投影。
    """
    N, T, _ = traj_uvz.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v, z = traj_uvz[..., 0], traj_uvz[..., 1], traj_uvz[..., 2]
    valid_z = (z > 0.01) & (z < 50.0) & np.isfinite(z)
    x_cam = np.where(valid_z, (u - cx) * z / (fx + 1e-8), np.nan)
    y_cam = np.where(valid_z, (v - cy) * z / (fy + 1e-8), np.nan)
    z_safe = np.where(valid_z, z, np.nan)
    pts_cam = np.stack([x_cam, y_cam, z_safe], axis=-1)  # (N, T, 3)
    ones = np.ones((N, T, 1), dtype=pts_cam.dtype)
    pts_cam_h = np.concatenate([pts_cam, ones], axis=-1)  # (N, T, 4)
    c2w = np.linalg.inv(w2c)
    pts_world = (c2w @ pts_cam_h.reshape(-1, 4).T).T.reshape(N, T, 4)
    out = pts_world[..., :3].astype(np.float32)
    out[~valid_z] = np.nan
    return out


def load_main_npz_for_dense(main_npz_path, downsample=4):
    """
    从主 NPZ 加载密集点云数据（类似 SpaTrackerV2）

    时域选取说明（infer 数据流）：
    - 主 NPZ 的 coords/depths 来自首段 query_frame_results[0]，是推理的**原始输出**，
      未经过 retarget_trajectories。形状为 (T_segment, N, 3)，T_segment = 段内视频帧数。
    - sample NPZ 在保存前会对 traj 做 retarget_trajectories，插值到 128 步（弧长均匀），
      因此 sample NPZ 的时域与视频帧不对齐。
    - 主 NPZ 的 coords 与 depths 一一对应视频帧，可直接用于帧对齐的密集点云可视化。

    返回: (dense_points_per_frame, dense_colors_per_frame, keypoint_traj, n_frames)
    - dense_points_per_frame: list of (M, 3) 每帧的密集点云
    - dense_colors_per_frame: list of (M, 3) 每帧的 RGB 颜色 [0,1]
    - keypoint_traj: (N, T, 3) 与视频帧对齐的 keypoint 轨迹
    """
    from PIL import Image

    data = np.load(main_npz_path)
    coords = data["coords"]  # (T, N, 3)
    depths = data["depths"].astype(np.float32)  # (T, H, W)
    intrinsics = data["intrinsics"]  # (T, 3, 3)
    extrinsics = data["extrinsics"]  # (T, 4, 4) w2c

    # 检查时间维度是否一致
    T_coords = coords.shape[0]
    T_depths = depths.shape[0]
    T_intr = intrinsics.shape[0]
    T_extr = extrinsics.shape[0]
    T = min(T_coords, T_depths, T_intr, T_extr)

    if not (T_coords == T_depths == T_intr == T_extr):
        logger.warning(
            f"[load_main_npz_for_dense] Time length mismatch: "
            f"coords={T_coords}, depths={T_depths}, intr={T_intr}, extr={T_extr}; "
            f"using first {T} frames for dense pointcloud."
        )
        coords = coords[:T]
        depths = depths[:T]
        intrinsics = intrinsics[:T]
        extrinsics = extrinsics[:T]

    T, N, _ = coords.shape
    H, W = depths.shape[1], depths.shape[2]
    intrinsics = intrinsics[:T]  # 对齐到 T 帧
    extrinsics = extrinsics[:T]  # 对齐到 T 帧
    c2w = np.linalg.inv(extrinsics)  # (T, 4, 4)

    # 加载 RGB：优先 NPZ 内 video，否则从 images 文件夹按帧加载
    rgb_frames = None
    if "video" in data:
        rgb_frames = data["video"]  # (T, H, W, 3) [0,1]
    data.close()

    if rgb_frames is None:
        images_dir = Path(main_npz_path).parent / "images"
        video_name = Path(main_npz_path).stem
        available = {}
        for f in images_dir.glob(f"{video_name}_*.png"):
            try:
                idx = int(f.stem.split("_")[-1])
                available[idx] = str(f)
            except ValueError:
                pass
        if available:
            frame_indices = np.array(sorted(available.keys()))

            def load_and_norm(path):
                img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
                if img.shape[0] != H or img.shape[1] != W:
                    img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((W, H), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0
                return img

            rgb_cache = {idx: load_and_norm(available[idx]) for idx in frame_indices}
            rgb_frames = np.array([rgb_cache[int(frame_indices[np.argmin(np.abs(frame_indices - t))])] for t in range(T)], dtype=np.float32)
        else:
            rgb_frames = np.ones((T, H, W, 3), dtype=np.float32) * 0.5

    # Keypoint 轨迹：转置为 (N, T, 3)，与视频帧数一致
    keypoint_traj = np.transpose(coords, (1, 0, 2))  # (N, T, 3)

    # 深度反投影得到密集点云，并采样对应 RGB
    depth_batch = depths[:, None, :, :]  # (T, 1, H, W)
    K_batch = intrinsics
    xyz = unproject_by_depth(depth_batch, K_batch, c2w)  # (T, 3, H, W)
    xyz = xyz.transpose(0, 2, 3, 1)  # (T, H, W, 3)

    dense_per_frame = []
    dense_colors_per_frame = []
    for t in range(T):
        pts = xyz[t]  # (H, W, 3)
        rgb = rgb_frames[t]  # (H, W, 3)
        pts_ds = pts[::downsample, ::downsample].reshape(-1, 3)
        rgb_ds = rgb[::downsample, ::downsample].reshape(-1, 3)
        valid = (pts_ds[:, 2] > 0) & (pts_ds[:, 2] < 10.0) & np.isfinite(pts_ds).all(axis=1)
        dense_per_frame.append(pts_ds[valid])
        dense_colors_per_frame.append(rgb_ds[valid].astype(np.float32))

    return dense_per_frame, dense_colors_per_frame, keypoint_traj, T


def load_dense_for_query_frame(episode_dir, video_name, frame_idx, main_npz_path, downsample=4):
    """
    为非首帧加载单帧密集点云：用 depth_raw + 主 NPZ 相机参数反投影。
    返回: (dense_points, dense_colors) 或 None
    """
    from PIL import Image

    depth_dir = episode_dir / "depth"
    images_dir = episode_dir / "images"
    depth_raw_path = depth_dir / f"{video_name}_{frame_idx}_raw.npz"
    if not depth_raw_path.exists():
        return None
    data = np.load(depth_raw_path)
    depth = data["depth"].astype(np.float32)
    data.close()

    main_data = np.load(main_npz_path)
    if frame_idx >= len(main_data["extrinsics"]) or frame_idx >= len(main_data["intrinsics"]):
        main_data.close()
        return None
    K = main_data["intrinsics"][frame_idx]
    w2c = main_data["extrinsics"][frame_idx]
    main_data.close()

    H, W = depth.shape
    c2w = np.linalg.inv(w2c)
    depth_batch = depth[None, None, :, :]  # (1, 1, H, W)
    K_batch = K[None, :, :]
    c2w_batch = c2w[None, :, :]
    xyz = unproject_by_depth(depth_batch, K_batch, c2w_batch)
    xyz = xyz[0].transpose(1, 2, 0)  # (H, W, 3)

    pts_ds = xyz[::downsample, ::downsample].reshape(-1, 3)
    valid = (pts_ds[:, 2] > 0) & (pts_ds[:, 2] < 10.0) & np.isfinite(pts_ds).all(axis=1)
    dense_points = pts_ds[valid]

    img_path = images_dir / f"{video_name}_{frame_idx}.png"
    if img_path.exists():
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((W, H), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0
        rgb_ds = rgb[::downsample, ::downsample].reshape(-1, 3)
        dense_colors = rgb_ds[valid].astype(np.float32)
    else:
        dense_colors = np.tile([0.4, 0.7, 0.9], (len(dense_points), 1)).astype(np.float32)

    return dense_points, dense_colors


def normalize_to_first_frame(traj, extrinsics_first):
    """
    将轨迹变换到首帧相机坐标系（类似 SpaTrackerV2）
    traj: (N, T, 3) 世界坐标
    extrinsics_first: (4, 4) w2c
    """
    N, T, _ = traj.shape
    ones = np.ones((N, T, 1), dtype=traj.dtype)
    traj_h = np.concatenate([traj, ones], axis=-1)  # (N, T, 4)
    traj_cam = (extrinsics_first @ traj_h.transpose(0, 1, 2).reshape(-1, 4).T).T
    return traj_cam.reshape(N, T, 4)[:, :, :3]


def compute_motion_rank(traj_sub):
    """
    计算每个 keypoint 的运动量（轨迹弧长），返回按运动量降序排列的索引。
    traj_sub: (N, T, 3)
    返回: order (N,), motion (N,) - 运动量从大到小
    """
    valid = np.isfinite(traj_sub).all(axis=-1)  # (N, T)
    d = np.diff(traj_sub, axis=1)  # (N, T-1, 3)
    valid_pair = valid[:, :-1] & valid[:, 1:]
    d_norm = np.where(valid_pair, np.linalg.norm(d, axis=-1), 0.0)
    motion = np.sum(d_norm, axis=1)
    return np.argsort(-motion), motion


def get_track_colors(pts, colormap="turbo"):
    """为轨迹点分配颜色"""
    pts_flat = pts.reshape(-1, 3)
    valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 1e10).all(axis=1)
    if not np.any(valid):
        return np.ones((len(pts_flat), 3)) * 0.5
    mins = np.nanmin(pts_flat[valid], axis=0)
    maxs = np.nanmax(pts_flat[valid], axis=0)
    if np.all(maxs == mins):
        maxs = mins + 1
    pts_norm = (pts_flat - mins) / (maxs - mins)
    pts_norm = np.nan_to_num(pts_norm, nan=0.5, posinf=1, neginf=0)
    orders = np.argsort(np.argsort(np.sum(pts_norm**2, axis=1))) / max(len(pts_norm) - 1, 1)
    return np.array([colormaps[colormap](o)[:3] for o in orders])


def main():
    parser = argparse.ArgumentParser(description="3D Keypoint 动画可视化")
    parser.add_argument(
        "--episode_dir",
        type=str,
        required=True,
        help="episode 目录，如 output_bridge_depth_grid80/00000/images0",
    )
    parser.add_argument(
        "--query_frame",
        type=int,
        default=None,
        help="指定查询帧索引，默认使用第一个可用的",
    )
    parser.add_argument(
        "--keypoint_stride",
        type=int,
        default=10,
        help="每 N 个 keypoint 显示 1 个；减小可显示更多（如 --keypoint_stride 2 得 3200 点），增大可提升性能（默认 10，6400->640）",
    )
    parser.add_argument(
        "--dense_pointcloud",
        action="store_true",
        help="显示密集点云：首帧用主 NPZ 多帧；非首帧用 depth_raw 单帧（静态）",
    )
    parser.add_argument(
        "--dense_downsample",
        type=int,
        default=4,
        help="密集点云下采样因子（默认 4，减小可提高密度但更卡）",
    )
    parser.add_argument(
        "--normalize_camera",
        action="store_true",
        help="将轨迹变换到首帧相机坐标系（类似 SpaTrackerV2）",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    samples_dir = episode_dir / "samples"
    video_name = episode_dir.name  # e.g. images0
    main_npz = episode_dir / f"{video_name}.npz"

    # 首帧 + 密集点云：使用主 NPZ 多帧；非首帧走 sample 分支并用 depth_raw 单帧
    use_dense_mode = (
        args.dense_pointcloud
        and main_npz.exists()
        and (args.query_frame is None or args.query_frame == 0)
    )
    if use_dense_mode:
        logger.info("使用主 NPZ 加载（密集点云 + 帧对齐轨迹）")
        dense_per_frame, dense_colors_per_frame, keypoint_traj, n_valid = load_main_npz_for_dense(
            main_npz, downsample=args.dense_downsample
        )
        n_total = keypoint_traj.shape[0]
        stride = max(1, args.keypoint_stride)
        if n_total > 500 and stride > 1:
            stride = 1
            logger.info(f"Keypoints 总数 {n_total} > 500，默认 stride 设为 1 以显示全部")
        indices = np.arange(0, n_total, stride)
        traj_sub = keypoint_traj[indices]  # (n_show, T, 3)
        n_show = len(indices)
        traj_full = keypoint_traj
        valid_steps = np.ones(n_valid, dtype=bool)
        frame_idx = 0
        logger.info(f"Keypoints: {n_total} -> {n_show}, 视频帧数: {n_valid}, 密集点云: 每帧 ~{len(dense_per_frame[0])} 点")
    else:
        dense_colors_per_frame = None
        if not samples_dir.exists():
            logger.error(f"samples 目录不存在: {samples_dir}")
            return
        sample_files = sorted(samples_dir.glob(f"{video_name}_*.npz"))
        if not sample_files:
            logger.error(f"未找到 sample 文件: {samples_dir}/{video_name}_*.npz")
            return
        query_frames = []
        for f in sample_files:
            stem = f.stem
            idx_str = stem.split("_")[-1]
            try:
                query_frames.append((int(idx_str), str(f)))
            except ValueError:
                continue
        if not query_frames:
            logger.error("无法解析查询帧索引")
            return
        query_frames.sort(key=lambda x: x[0])
        if args.query_frame is not None:
            match = [qf for qf in query_frames if qf[0] == args.query_frame]
            frame_idx, npz_path = match[0] if match else query_frames[0]
        else:
            frame_idx, npz_path = query_frames[0]
        logger.info(f"加载: {npz_path} (查询帧 {frame_idx})")
        traj, valid_steps, keypoints, _ = load_sample_npz(npz_path)
        n_valid = int(np.sum(valid_steps))
        # sample NPZ 的 traj 可能为 (u_pixel, v_pixel, z_depth)，需转为世界坐标
        valid_pts = traj[np.isfinite(traj).all(axis=-1)]
        is_pixel_format = len(valid_pts) > 0 and (np.abs(valid_pts[:, 0]).max() > 200 or np.abs(valid_pts[:, 1]).max() > 200)
        if is_pixel_format and main_npz.exists():
            main_data = np.load(main_npz)
            K = main_data["intrinsics"]
            w2c = main_data["extrinsics"]
            if frame_idx < len(K) and frame_idx < len(w2c):
                traj = traj_pixel_depth_to_world(traj, K[frame_idx], w2c[frame_idx])
                logger.info("已将 traj 从 (u,v,z) 转为世界坐标")
            main_data.close()
        traj_full = np.full_like(traj, np.nan)
        traj_full[:, valid_steps, :] = traj[:, valid_steps, :]
        n_total = traj_full.shape[0]
        # 当 NPZ 中 keypoint 很多（如 6400）时，默认 stride=1 以显示全部，避免误以为“最多只能显示 640”
        stride = max(1, args.keypoint_stride)
        if n_total > 500 and stride > 1:
            stride = 1
            logger.info(f"Keypoints 总数 {n_total} > 500，默认 stride 设为 1 以显示全部")
        indices = np.arange(0, n_total, stride)
        traj_sub = traj_full[indices]
        n_show = len(indices)
        dense_per_frame = None
        dense_colors_per_frame = None
        if args.dense_pointcloud and main_npz.exists():
            result = load_dense_for_query_frame(
                episode_dir, video_name, frame_idx, main_npz, args.dense_downsample
            )
            if result is not None:
                dense_pts, dense_cols = result
                dense_per_frame = [dense_pts]
                dense_colors_per_frame = [dense_cols]
                logger.info(f"非首帧密集点云: 查询帧 {frame_idx}, ~{len(dense_pts)} 点（静态）")
        logger.info(f"Keypoints: {n_total} -> {n_show} (stride={stride}), 有效帧: {n_valid}")

    # 可选：归一化到首帧相机
    if args.normalize_camera and main_npz.exists():
        main_data = np.load(main_npz)
        extrs = main_data["extrinsics"]
        seg_start = frame_idx if frame_idx < len(extrs) else 0
        if seg_start < len(extrs):
            traj_sub = normalize_to_first_frame(traj_sub, extrs[seg_start])
            if dense_per_frame is not None:
                c2w = np.linalg.inv(extrs[seg_start])
                w2c = extrs[seg_start]
                for t in range(len(dense_per_frame)):
                    pts = dense_per_frame[t]
                    ones = np.ones((len(pts), 1))
                    pts_h = np.hstack([pts, ones])
                    dense_per_frame[t] = (w2c @ pts_h.T).T[:, :3]
        main_data.close()

    # 颜色：每个 keypoint 一个颜色，用首帧位置计算
    colors = np.asarray(get_track_colors(traj_sub[:, 0, :]), dtype=np.float32)  # (n_show, 3)

    # 预计算运动量排序（用于仅显示动态轨迹的优化）
    motion_order, motion_scores = compute_motion_rank(traj_sub)

    # Viser 服务器
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")

    # 点云句柄（用于动画更新）；stride 变化时需重建以支持任意点数
    points_0 = traj_sub[:, 0, :]
    valid_0 = np.isfinite(points_0).all(axis=1)
    pts_init = np.where(valid_0[:, None], points_0, np.zeros_like(points_0))
    point_cloud_handle = server.scene.add_point_cloud(
        name="keypoints",
        points=pts_init,
        colors=colors,
        point_size=0.03,
        point_shape="rounded",
        precision="float32",
    )

    def _rebuild_keypoint_pointcloud():
        """stride 变化时重建点云，确保 points/colors 数量一致"""
        nonlocal point_cloud_handle
        try:
            point_cloud_handle.remove()
        except KeyError:
            pass
        pts = get_points_at_time(int(gui_time.value))
        point_cloud_handle = server.scene.add_point_cloud(
            name="keypoints",
            points=pts,
            colors=(np.clip(colors * 255.0, 0, 255)).astype(np.uint8),
            point_size=gui_point_size.value,
            point_shape="rounded",
            precision="float32",
        )

    # 密集点云句柄（首帧+--dense_pointcloud 时）
    dense_point_cloud_handle = None
    if dense_per_frame is not None:
        d0 = dense_per_frame[0]
        c0 = dense_colors_per_frame[0] if dense_colors_per_frame is not None else np.tile([0.4, 0.7, 0.9], (len(d0), 1)).astype(np.float32)
        dense_point_cloud_handle = server.scene.add_point_cloud(
            name="dense_pointcloud",
            points=d0,
            colors=c0.astype(np.float32),
            point_size=0.015,
            point_shape="rounded",
        )

    with server.gui.add_folder("3D Keypoint 动画"):
        gui_time = server.gui.add_slider(
            "时间步",
            min=0,
            max=max(1, n_valid - 1),
            step=1,
            initial_value=0,
        )
        gui_playing = server.gui.add_checkbox("播放", True)
        gui_fps = server.gui.add_slider("帧率", min=1, max=60, step=1, initial_value=10)
        gui_keypoint_stride = server.gui.add_slider(
            "Keypoint 采样步长（1=全部）", min=1, max=min(100, max(20, n_total // 10)), step=1, initial_value=stride
        )
        gui_keypoint_count = server.gui.add_number("当前显示 Keypoint 数", initial_value=n_show, disabled=True)
        gui_keypoint_total = server.gui.add_number("NPZ 总 Keypoint 数", initial_value=n_total, disabled=True)
        gui_point_size = server.gui.add_slider(
            "点大小", min=0.001, max=2.0, step=0.005, initial_value=0.03
        )
        gui_dense_point_size = server.gui.add_slider(
            "密集点云大小", min=0.001, max=0.1, step=0.001, initial_value=0.015
        ) if dense_per_frame is not None else None
        gui_show_keypoints = server.gui.add_checkbox("显示 Keypoints", True)
        gui_show_dense = server.gui.add_checkbox("显示密集点云", True) if dense_per_frame is not None else None
        gui_show_trails = server.gui.add_checkbox("显示轨迹线", False)
        gui_trail_full = server.gui.add_checkbox("完整轨迹（显示整段 0→末帧）", True)
        gui_trail_line_width = server.gui.add_slider("轨迹线宽", min=0.5, max=15.0, step=0.5, initial_value=4.0)
        gui_trail_dynamic_only = server.gui.add_checkbox("仅动态轨迹（性能优化）", True)
        gui_trail_dynamic_ratio = server.gui.add_slider("动态比例", min=0.05, max=1.0, step=0.05, initial_value=0.2)

    trail_handles = []
    _trail_name_counter = [0]

    def apply_keypoint_stride():
        nonlocal traj_sub, n_show, colors, motion_order
        s = max(1, int(gui_keypoint_stride.value))
        indices = np.arange(0, n_total, s)
        traj_sub = traj_full[indices]
        n_show = len(indices)
        if gui_keypoint_count is not None:
            gui_keypoint_count.value = n_show
        colors = np.asarray(get_track_colors(traj_sub[:, 0, :]), dtype=np.float32)
        motion_order, _ = compute_motion_rank(traj_sub)
        _rebuild_keypoint_pointcloud()
        update_display()
        update_trails()

    def get_points_at_time(t):
        """获取时间 t 的点位置"""
        t = min(t, n_valid - 1)
        t = min(t, traj_sub.shape[1] - 1)
        if t < 0:
            t = 0
        pts = traj_sub[:, t, :].copy()
        invalid = ~np.isfinite(pts).any(axis=1)
        pts[invalid] = 0.0
        return pts

    def update_display():
        t = int(gui_time.value)
        t = min(t, n_valid - 1)
        pts = get_points_at_time(t)
        point_cloud_handle.points = pts
        point_cloud_handle.colors = (np.clip(colors * 255.0, 0, 255)).astype(np.uint8)
        point_cloud_handle.point_size = gui_point_size.value
        point_cloud_handle.visible = gui_show_keypoints.value
        if dense_point_cloud_handle is not None and dense_per_frame is not None:
            # 非首帧密集点云仅一帧（静态），首帧为多帧
            dense_idx = 0 if len(dense_per_frame) == 1 else t
            dense_point_cloud_handle.points = dense_per_frame[dense_idx]
            n_d = len(dense_per_frame[dense_idx])
            if dense_colors_per_frame is not None:
                dense_point_cloud_handle.colors = dense_colors_per_frame[dense_idx].astype(np.float32)
            else:
                dense_point_cloud_handle.colors = np.tile([0.4, 0.7, 0.9], (n_d, 1)).astype(np.float32)
            if gui_dense_point_size is not None:
                dense_point_cloud_handle.point_size = gui_dense_point_size.value
            dense_point_cloud_handle.visible = gui_show_dense.value if gui_show_dense is not None else True

    def update_trails():
        for h in trail_handles:
            try:
                h.remove()
            except KeyError:
                pass
        trail_handles.clear()
        if not gui_show_trails.value:
            return
        t = int(gui_time.value)
        t_end = n_valid - 1 if gui_trail_full.value else t
        lw = gui_trail_line_width.value

        # 仅动态轨迹：按运动量取 top K
        if gui_trail_dynamic_only.value:
            n_dynamic = max(10, int(n_show * gui_trail_dynamic_ratio.value))
            indices = motion_order[:n_dynamic]
        else:
            indices = np.arange(n_show)

        # 确保不越界：indices 不超过 traj_sub 行数，j+1 不超过帧数（每次用前再读一次，避免与 stride 回调竞态）
        n_pts = traj_sub.shape[0]
        n_frames = traj_sub.shape[1]
        indices = indices[indices < n_pts]
        t_end = min(t_end, n_frames - 1)
        if t_end < 1:
            return

        # 批量收集所有有效线段，单次 add_line_segments 减少句柄数量
        all_segs = []
        all_cols = []
        for i in indices:
            if i >= traj_sub.shape[0]:
                continue
            for j in range(t_end):
                if j + 1 >= traj_sub.shape[1]:
                    break
                p0 = traj_sub[i, j, :]
                p1 = traj_sub[i, j + 1, :]
                if np.isfinite(p0).all() and np.isfinite(p1).all():
                    all_segs.append([p0, p1])
                    all_cols.append([colors[i], colors[i]])

        if all_segs:
            segs_arr = np.array(all_segs)
            cols_arr = np.array(all_cols)
            _trail_name_counter[0] += 1
            h = server.scene.add_line_segments(
                name=f"trails_batched_{_trail_name_counter[0]}",
                points=segs_arr,
                colors=cols_arr,
                line_width=lw,
            )
            trail_handles.append(h)

    @gui_time.on_update
    def _(_):
        update_display()
        update_trails()

    @gui_fps.on_update
    def _(_):
        pass

    @gui_keypoint_stride.on_update
    def _(_):
        apply_keypoint_stride()

    @gui_point_size.on_update
    def _(_):
        point_cloud_handle.point_size = gui_point_size.value

    @gui_show_keypoints.on_update
    def _(_):
        point_cloud_handle.visible = gui_show_keypoints.value

    @gui_show_trails.on_update
    def _(_):
        update_trails()

    @gui_trail_full.on_update
    def _(_):
        update_trails()

    @gui_trail_line_width.on_update
    def _(_):
        update_trails()

    @gui_trail_dynamic_only.on_update
    def _(_):
        update_trails()

    @gui_trail_dynamic_ratio.on_update
    def _(_):
        update_trails()

    if gui_dense_point_size is not None:

        @gui_dense_point_size.on_update
        def _(_):
            if dense_point_cloud_handle is not None:
                dense_point_cloud_handle.point_size = gui_dense_point_size.value

    if gui_show_dense is not None:

        @gui_show_dense.on_update
        def _(_):
            if dense_point_cloud_handle is not None:
                dense_point_cloud_handle.visible = gui_show_dense.value

    update_display()

    logger.info(f"Viser 服务器: http://localhost:{args.port}")
    logger.info("使用滑块或勾选「播放」查看 3D keypoint 动画")

    # 动画循环（类似 viser_utils）
    try:
        while True:
            if gui_playing.value and n_valid > 1:
                new_t = (int(gui_time.value) + 1) % n_valid
                gui_time.value = new_t
            time.sleep(1.0 / max(1, gui_fps.value))
    except KeyboardInterrupt:
        logger.info("退出")


if __name__ == "__main__":
    main()
