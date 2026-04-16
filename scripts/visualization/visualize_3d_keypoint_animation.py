#!/usr/bin/env python3
"""
TraceForge 推理结果的 3D Keypoint 动画可视化。

v2 布局：
- 总是从 `samples/<video>_<query>.npz` 读取轨迹；
- `--dense_pointcloud` 时，按 `segment_frame_indices` 从 scene cache 或 source refs
  逐帧重建动态密集点云；
- 不再依赖“首帧专用主 NPZ”。

legacy 布局：
- 仍支持旧 sample/main NPZ；
- `query_frame=0 + --dense_pointcloud` 保留主 NPZ 的动态 dense fallback；
- 非首帧 dense 仍退化为单帧静态背景。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import viser
from matplotlib import colormaps
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import (
    LEGACY_LAYOUT,
    RENDER_MODE_FINITE,
    RENDER_MODE_HYBRID,
    RENDER_MODE_SUPERVISION,
    RENDER_MODES,
    SceneReader,
    build_pointcloud_from_frame,
    build_sample_visualization_view,
    ensure_uint8_video,
    list_sample_query_frames,
    normalize_sample_data,
    traj_uvz_to_world,
)


def normalize_to_first_frame(traj: np.ndarray, extrinsics_first: np.ndarray) -> np.ndarray:
    if len(traj) == 0:
        return traj
    ones = np.ones((*traj.shape[:2], 1), dtype=traj.dtype)
    traj_h = np.concatenate([traj, ones], axis=-1)
    traj_cam = (extrinsics_first @ traj_h.reshape(-1, 4).T).T.reshape(*traj.shape[:2], 4)
    return traj_cam[..., :3]


def compute_motion_rank(traj_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(traj_sub).all(axis=-1)
    d = np.diff(traj_sub, axis=1)
    valid_pair = valid[:, :-1] & valid[:, 1:]
    d_norm = np.where(valid_pair, np.linalg.norm(d, axis=-1), 0.0)
    motion = np.sum(d_norm, axis=1)
    return np.argsort(-motion), motion


def get_track_colors(pts: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    pts_flat = pts.reshape(-1, 3)
    valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat) < 1e10).all(axis=1)
    if not np.any(valid):
        return np.ones((len(pts), 3), dtype=np.float32) * 0.5
    mins = np.nanmin(pts_flat[valid], axis=0)
    maxs = np.nanmax(pts_flat[valid], axis=0)
    if np.all(maxs == mins):
        maxs = mins + 1
    pts_norm = (pts - mins) / (maxs - mins)
    pts_norm = np.nan_to_num(pts_norm, nan=0.5, posinf=1, neginf=0)
    score = np.sum(pts_norm[:, 0, :] ** 2, axis=1)
    order = np.argsort(np.argsort(score)) / max(len(score) - 1, 1)
    return np.asarray([colormaps[colormap](float(v))[:3] for v in order], dtype=np.float32)


def fade_track_colors(colors: np.ndarray, *, blend: float = 0.72) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    blend = float(np.clip(blend, 0.0, 1.0))
    return colors * (1.0 - blend) + blend


def colors_to_uint8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        return colors
    if colors.size == 0:
        return np.zeros(colors.shape, dtype=np.uint8)
    return np.clip(colors * 255.0, 0, 255).astype(np.uint8)


def compute_scene_bounds(point_sets: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
    finite_sets: list[np.ndarray] = []
    for points in point_sets:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) > 0:
            finite_sets.append(finite)

    if not finite_sets:
        return None

    merged = np.concatenate(finite_sets, axis=0)
    mins = np.min(merged, axis=0)
    maxs = np.max(merged, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.linalg.norm(maxs - mins))
    radius = max(radius, 0.05)
    return center.astype(np.float64), radius


def set_initial_camera_from_scene(
    server: viser.ViserServer,
    *,
    scene_center: np.ndarray,
    scene_radius: float,
) -> None:
    scene_center = np.asarray(scene_center, dtype=np.float64)
    direction = np.asarray([2.2, 1.4, 2.2], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    distance = max(scene_radius * 4.0, 0.35)

    server.initial_camera.look_at = scene_center
    server.initial_camera.position = scene_center + direction * distance
    server.initial_camera.near = max(scene_radius * 0.02, 1e-3)
    server.initial_camera.far = max(scene_radius * 40.0, 5.0)

    logger.info(
        "Initial camera fitted to scene: "
        f"center={scene_center.tolist()}, radius={scene_radius:.4f}, distance={distance:.4f}"
    )


def load_dense_sequence_from_scene(
    scene_reader: SceneReader,
    *,
    frame_indices: np.ndarray,
    downsample: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
    dense_per_frame: list[np.ndarray] = []
    dense_colors_per_frame: list[np.ndarray] = []
    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)
        depth = scene_reader.get_depth_frame(frame_idx)
        rgb = scene_reader.get_rgb_frame(frame_idx)
        dense_points, dense_colors = build_pointcloud_from_frame(
            depth=depth,
            rgb=rgb,
            intrinsics=intrinsics_all[frame_idx],
            w2c=extrinsics_all[frame_idx],
            downsample=downsample,
        )
        dense_per_frame.append(dense_points)
        dense_colors_per_frame.append(dense_colors)
    return dense_per_frame, dense_colors_per_frame


def load_main_npz_for_dense(main_npz_path: Path, downsample: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    from PIL import Image

    data = np.load(main_npz_path)
    try:
        coords = data["coords"].astype(np.float32)
        depths = data["depths"].astype(np.float32)
        intrinsics = data["intrinsics"].astype(np.float32)
        extrinsics = data["extrinsics"].astype(np.float32)
        num_frames = min(len(coords), len(depths), len(intrinsics), len(extrinsics))
        coords = coords[:num_frames]
        depths = depths[:num_frames]
        intrinsics = intrinsics[:num_frames]
        extrinsics = extrinsics[:num_frames]

        rgb_frames = None
        if "video" in data:
            rgb_frames = np.asarray(data["video"])
            if rgb_frames.ndim == 4 and rgb_frames.shape[1] in (1, 3):
                rgb_frames = rgb_frames.transpose(0, 2, 3, 1)
            rgb_frames = ensure_uint8_video(rgb_frames).astype(np.float32) / 255.0
    finally:
        data.close()

    if rgb_frames is None:
        images_dir = main_npz_path.parent / "images"
        video_name = main_npz_path.stem
        rgb_frames = []
        for frame_idx in range(num_frames):
            image_path = images_dir / f"{video_name}_{frame_idx}.png"
            if image_path.is_file():
                rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0
            else:
                h, w = depths[frame_idx].shape
                rgb = np.full((h, w, 3), 0.5, dtype=np.float32)
            rgb_frames.append(rgb)
        rgb_frames = np.stack(rgb_frames, axis=0)

    dense_per_frame: list[np.ndarray] = []
    dense_colors_per_frame: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        dense_points, dense_colors = build_pointcloud_from_frame(
            depth=depths[frame_idx],
            rgb=rgb_frames[frame_idx],
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics[frame_idx],
            downsample=downsample,
        )
        dense_per_frame.append(dense_points)
        dense_colors_per_frame.append(dense_colors)

    keypoint_traj = np.transpose(coords, (1, 0, 2))
    return dense_per_frame, dense_colors_per_frame, keypoint_traj


def load_static_dense_for_legacy_frame(
    scene_reader: SceneReader,
    *,
    frame_idx: int,
    downsample: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dense_points, dense_colors = load_dense_sequence_from_scene(
        scene_reader,
        frame_indices=np.array([frame_idx], dtype=np.int32),
        downsample=downsample,
    )
    return dense_points, dense_colors


def resolve_query_frame(episode_dir: Path, query_frame: int | None) -> tuple[int, Path]:
    video_name = episode_dir.name
    available = list_sample_query_frames(episode_dir, video_name)
    if not available:
        raise FileNotFoundError(f"No sample NPZ files found under {episode_dir / 'samples'}")

    if query_frame is None:
        query_frame = available[0]
    elif query_frame not in available:
        logger.warning(
            f"query_frame={query_frame} not found, fallback to {available[0]}; available={available}"
        )
        query_frame = available[0]

    sample_path = episode_dir / "samples" / f"{video_name}_{query_frame}.npz"
    return int(query_frame), sample_path


def load_sample_world_trajectory(
    scene_reader: SceneReader,
    sample_path: Path,
) -> dict:
    sample = normalize_sample_data(sample_path)
    render_view = build_sample_visualization_view(sample, render_mode=RENDER_MODE_SUPERVISION)
    query_frame_idx = int(sample["query_frame_index"])
    intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
    query_intrinsics = intrinsics_all[query_frame_idx].astype(np.float32)
    query_w2c = extrinsics_all[query_frame_idx].astype(np.float32)

    traj_uvz_finite = np.asarray(render_view["traj_uvz_finite"], dtype=np.float32)
    traj_world = traj_uvz_to_world(traj_uvz_finite, query_intrinsics, query_w2c)

    logger.info(
        f"traj_valid_mask 过滤后轨迹数: {render_view['kept_num_tracks']}/{render_view['raw_num_tracks']}"
    )

    return {
        "traj_world": traj_world,
        "keypoints": np.asarray(render_view["keypoints"], dtype=np.float32),
        "query_frame_idx": query_frame_idx,
        "query_w2c": query_w2c,
        "segment_frame_indices": np.asarray(render_view["segment_frame_indices"], dtype=np.int32),
        "traj_supervision_mask": np.asarray(render_view["supervision_step_mask"], dtype=bool),
        "traj_finite_mask": np.asarray(render_view["finite_step_mask"], dtype=bool),
        "raw_num_tracks": int(render_view["raw_num_tracks"]),
        "filtered_num_tracks": int(render_view["kept_num_tracks"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Keypoint 动画可视化")
    parser.add_argument(
        "--episode_dir",
        type=str,
        required=True,
        help="camera 输出目录，例如 <episode>/trajectory/varied_camera_3 或外部 out_dir 下的对应目录",
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
        help="每 N 个 keypoint 显示 1 个；大于 1 可提升性能",
    )
    parser.add_argument(
        "--dense_pointcloud",
        action="store_true",
        help="v2: 任意查询帧都显示动态 dense pointcloud；legacy: 仅 query_frame=0 保持动态，其他帧退化为静态",
    )
    parser.add_argument(
        "--dense_downsample",
        type=int,
        default=4,
        help="密集点云下采样因子",
    )
    parser.add_argument(
        "--normalize_camera",
        action="store_true",
        help="将轨迹和 dense pointcloud 变换到查询帧相机坐标系",
    )
    parser.add_argument(
        "--dense_playback_stride",
        type=int,
        default=0,
        help="自动播放时密集点云每 N 帧刷新 1 次；0 表示按点数自动选择",
    )
    parser.add_argument(
        "--preload_playback",
        action="store_true",
        help="启动时预加载各帧 pointcloud，播放时仅切换可见性；可显著降低播放卡顿，但启动更慢、更占浏览器内存",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=RENDER_MODE_SUPERVISION,
        choices=RENDER_MODES,
        help="显示模式：supervision / finite / hybrid",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir).resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    video_name = episode_dir.name
    main_npz = episode_dir / f"{video_name}.npz"
    query_frame_idx, sample_path = resolve_query_frame(episode_dir, args.query_frame)

    dense_per_frame: list[np.ndarray] | None = None
    dense_colors_per_frame: list[np.ndarray] | None = None
    use_legacy_main_dense = False

    with SceneReader(episode_dir) as scene_reader:
        layout = scene_reader.layout
        sample_bundle = load_sample_world_trajectory(scene_reader, sample_path)
        traj_full = sample_bundle["traj_world"]
        query_frame_idx = sample_bundle["query_frame_idx"]
        query_w2c = sample_bundle["query_w2c"]
        segment_frame_indices = sample_bundle["segment_frame_indices"]
        supervision_full = sample_bundle["traj_supervision_mask"]
        finite_full = sample_bundle["traj_finite_mask"]

        if (
            args.dense_pointcloud
            and layout == LEGACY_LAYOUT
            and query_frame_idx == 0
            and main_npz.is_file()
        ):
            logger.info("legacy/query_frame=0: 使用主 NPZ 动态 dense fallback")
            dense_per_frame, dense_colors_per_frame, keypoint_traj = load_main_npz_for_dense(
                main_npz, downsample=args.dense_downsample
            )
            sample = normalize_sample_data(sample_path)
            render_view = build_sample_visualization_view(sample, render_mode=RENDER_MODE_SUPERVISION)
            traj_valid_mask = sample["traj_valid_mask"].astype(bool, copy=False)
            if len(traj_valid_mask) == keypoint_traj.shape[0]:
                keypoint_traj = keypoint_traj[traj_valid_mask]
            supervision_full = np.asarray(render_view["supervision_step_mask"], dtype=bool)
            if supervision_full.shape[1] > keypoint_traj.shape[1]:
                supervision_full = supervision_full[:, : keypoint_traj.shape[1]]
            finite_full = np.isfinite(keypoint_traj).all(axis=-1)
            traj_full = keypoint_traj.astype(np.float32)
            segment_frame_indices = np.arange(traj_full.shape[1], dtype=np.int32)
            use_legacy_main_dense = True
        elif args.dense_pointcloud and layout != LEGACY_LAYOUT:
            logger.info("v2: 使用 scene artifacts 重建动态 dense pointcloud")
            dense_per_frame, dense_colors_per_frame = load_dense_sequence_from_scene(
                scene_reader,
                frame_indices=segment_frame_indices,
                downsample=args.dense_downsample,
            )
        elif args.dense_pointcloud and layout == LEGACY_LAYOUT:
            logger.info("legacy/nonzero query_frame: dense pointcloud 退化为单帧静态背景")
            dense_per_frame, dense_colors_per_frame = load_static_dense_for_legacy_frame(
                scene_reader,
                frame_idx=query_frame_idx,
                downsample=args.dense_downsample,
            )

    n_total = int(traj_full.shape[0])
    n_valid = int(traj_full.shape[1])
    if supervision_full is None or supervision_full.shape != traj_full.shape[:2]:
        supervision_full = finite_full.copy()
    else:
        supervision_full = np.asarray(supervision_full, dtype=bool) & finite_full
    stride = max(1, args.keypoint_stride)

    if args.normalize_camera:
        traj_full = normalize_to_first_frame(traj_full, query_w2c)
        if dense_per_frame is not None:
            for idx, pts in enumerate(dense_per_frame):
                if len(pts) == 0:
                    continue
                ones = np.ones((len(pts), 1), dtype=np.float32)
                pts_h = np.hstack([pts.astype(np.float32), ones])
                dense_per_frame[idx] = (query_w2c @ pts_h.T).T[:, :3].astype(np.float32)
    if dense_colors_per_frame is not None:
        dense_colors_per_frame = [colors_to_uint8(frame_colors) for frame_colors in dense_colors_per_frame]

    def resolve_render_masks(render_mode: str) -> tuple[np.ndarray, np.ndarray]:
        if render_mode == RENDER_MODE_SUPERVISION:
            return supervision_full.copy(), np.zeros_like(supervision_full, dtype=bool)
        if render_mode == RENDER_MODE_FINITE:
            return finite_full.copy(), np.zeros_like(finite_full, dtype=bool)
        if render_mode == RENDER_MODE_HYBRID:
            primary_full = supervision_full.copy()
            secondary_full = finite_full & (~primary_full)
            return primary_full, secondary_full
        raise ValueError(f"Unsupported render_mode: {render_mode}")

    indices = np.arange(0, n_total, stride)
    traj_sub = traj_full[indices]
    render_primary_full, render_secondary_full = resolve_render_masks(str(args.render_mode))
    render_primary_sub = render_primary_full[indices]
    render_secondary_sub = render_secondary_full[indices]
    n_show = len(indices)
    colors = get_track_colors(
        traj_sub[:, :1, :] if traj_sub.shape[1] > 0 else np.zeros((n_show, 1, 3), dtype=np.float32)
    )
    ghost_colors = fade_track_colors(colors)
    motion_order, _motion_scores = compute_motion_rank(traj_sub)
    frame_point_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    logger.info(
        f"加载 {sample_path.name}: query_frame={query_frame_idx}, "
        f"layout={layout}, keypoints={n_total}->{n_show}, segment_len={n_valid}, render_mode={args.render_mode}"
    )
    dense_point_count = 0
    if dense_per_frame is not None:
        mode = "dynamic" if len(dense_per_frame) > 1 else "static"
        dense_point_count = len(dense_per_frame[0]) if dense_per_frame else 0
        logger.info(
            f"dense pointcloud: {mode}, frames={len(dense_per_frame)}, "
            f"first_frame_points={dense_point_count}"
        )
    if use_legacy_main_dense:
        logger.info("legacy 主 NPZ dense 模式下，轨迹来自主 NPZ 原始帧对齐坐标")

    resolved_dense_playback_stride = max(1, int(args.dense_playback_stride))
    if args.dense_playback_stride <= 0:
        resolved_dense_playback_stride = 1
        if dense_per_frame is not None and len(dense_per_frame) > 1:
            if dense_point_count >= 40000:
                resolved_dense_playback_stride = 2
            if dense_point_count >= 80000:
                resolved_dense_playback_stride = 4
    resolved_playback_keypoint_multiplier = 1
    if n_total >= 4000:
        resolved_playback_keypoint_multiplier = 2
    if n_total >= 8000:
        resolved_playback_keypoint_multiplier = 4
    resolved_playback_frame_step = 1
    if dense_point_count >= 40000 or n_total >= 4000:
        resolved_playback_frame_step = 2
    if dense_point_count >= 80000 or n_total >= 8000:
        resolved_playback_frame_step = 4
    if dense_per_frame is not None and len(dense_per_frame) > 1:
        logger.info(
            f"dense playback stride={resolved_dense_playback_stride} "
            "(1 表示每帧刷新 dense，大于 1 可提升播放流畅度)"
        )
    if resolved_playback_keypoint_multiplier > 1 or resolved_playback_frame_step > 1:
        logger.info(
            "playback light mode: "
            f"keypoint_stride_multiplier={resolved_playback_keypoint_multiplier}, "
            f"frame_step={resolved_playback_frame_step}"
        )
    use_preload_playback = bool(args.preload_playback)
    if use_preload_playback:
        logger.info("preload playback 已启用：启动时预加载各帧 pointcloud，播放时仅切换可见性")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    scene_bounds = compute_scene_bounds(
        [traj_full.reshape(-1, 3)] + ([dense_per_frame[0]] if dense_per_frame is not None and len(dense_per_frame) > 0 else [])
    )
    if scene_bounds is not None:
        scene_center, scene_radius = scene_bounds
        set_initial_camera_from_scene(
            server,
            scene_center=scene_center,
            scene_radius=scene_radius,
        )

    def get_points_at_time(t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_valid <= 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, empty, empty
        t = min(max(int(t), 0), n_valid - 1)
        pts = traj_sub[:, t, :].copy()
        finite_mask = np.isfinite(pts).all(axis=1)
        primary_mask = finite_mask & render_primary_sub[:, t]
        secondary_mask = finite_mask & render_secondary_sub[:, t]
        return (
            pts[primary_mask],
            colors[primary_mask],
            pts[secondary_mask],
            ghost_colors[secondary_mask],
        )

    points_0, colors_0, ghost_points_0, ghost_colors_0 = get_points_at_time(0)
    primary_point_cloud_handle = None
    secondary_point_cloud_handle = None
    dense_point_cloud_handle = None
    if not use_preload_playback:
        primary_point_cloud_handle = server.scene.add_point_cloud(
            name="keypoints_primary",
            points=points_0,
            colors=(np.clip(colors_0 * 255.0, 0, 255)).astype(np.uint8),
            point_size=0.03,
            point_shape="rounded",
            precision="float32",
        )
        secondary_point_cloud_handle = server.scene.add_point_cloud(
            name="keypoints_secondary",
            points=ghost_points_0,
            colors=(np.clip(ghost_colors_0 * 255.0, 0, 255)).astype(np.uint8),
            point_size=0.024,
            point_shape="rounded",
            precision="float32",
        )

        if dense_per_frame is not None:
            dense_point_cloud_handle = server.scene.add_point_cloud(
                name="dense_pointcloud",
                points=dense_per_frame[0],
                colors=dense_colors_per_frame[0],
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
        gui_light_playback = server.gui.add_checkbox("播放轻量模式", True)
        gui_playback_keypoint_multiplier = server.gui.add_slider(
            "播放时额外 Keypoint 步长",
            min=1,
            max=16,
            step=1,
            initial_value=resolved_playback_keypoint_multiplier,
        )
        gui_playback_frame_step = server.gui.add_slider(
            "播放时间步长",
            min=1,
            max=min(8, max(2, n_valid)),
            step=1,
            initial_value=min(resolved_playback_frame_step, min(8, max(2, n_valid))),
        )
        gui_dense_playback_stride = (
            server.gui.add_slider(
                "播放时 dense 刷新步长",
                min=1,
                max=min(16, max(2, n_valid)),
                step=1,
                initial_value=resolved_dense_playback_stride,
            )
            if dense_per_frame is not None and len(dense_per_frame) > 1
            else None
        )
        gui_render_mode = server.gui.add_dropdown(
            "显示模式",
            RENDER_MODES,
            initial_value=str(args.render_mode),
        )
        gui_keypoint_stride = server.gui.add_slider(
            "Keypoint 采样步长（1=全部）",
            min=1,
            max=min(100, max(20, max(n_total, 1) // 10)),
            step=1,
            initial_value=stride,
        )
        gui_keypoint_count = server.gui.add_number("当前显示 Keypoint 数", initial_value=n_show, disabled=True)
        gui_keypoint_total = server.gui.add_number("NPZ 总 Keypoint 数", initial_value=n_total, disabled=True)
        gui_point_size = server.gui.add_slider(
            "点大小", min=0.001, max=2.0, step=0.005, initial_value=0.03
        )
        gui_dense_point_size = (
            server.gui.add_slider(
                "密集点云大小", min=0.001, max=0.1, step=0.001, initial_value=0.015
            )
            if dense_per_frame is not None
            else None
        )
        gui_show_keypoints = server.gui.add_checkbox("显示 Keypoints", True)
        gui_show_dense = (
            server.gui.add_checkbox("显示密集点云", True)
            if dense_per_frame is not None
            else None
        )
        gui_show_trails = server.gui.add_checkbox("显示轨迹线", False)
        gui_trail_full = server.gui.add_checkbox("完整轨迹（显示整段 0→末帧）", True)
        gui_trail_line_width = server.gui.add_slider("轨迹线宽", min=0.5, max=15.0, step=0.5, initial_value=4.0)
        gui_trail_dynamic_only = server.gui.add_checkbox("仅动态轨迹（性能优化）", True)
        gui_trail_dynamic_ratio = server.gui.add_slider("动态比例", min=0.05, max=1.0, step=0.05, initial_value=0.2)

    trail_handles = []
    trail_name_counter = [0]
    last_dense_frame = [-1]
    active_keypoint_stride = [stride]
    preloaded_primary_handles = []
    preloaded_secondary_handles = []
    preloaded_dense_handles = []
    last_visible_keypoint_frame = [-1]
    last_visible_dense_handle = [-1]

    def resolve_dense_frame_index(t: int, *, force_exact: bool = False) -> int:
        if dense_per_frame is None or len(dense_per_frame) == 0:
            return 0
        if len(dense_per_frame) == 1:
            return 0
        if force_exact or not gui_playing.value or gui_dense_playback_stride is None:
            return min(t, len(dense_per_frame) - 1)
        stride_value = max(1, int(gui_dense_playback_stride.value))
        return min((t // stride_value) * stride_value, len(dense_per_frame) - 1)

    def resolve_active_keypoint_stride() -> int:
        base_stride = max(1, int(gui_keypoint_stride.value))
        if gui_playing.value and gui_light_playback.value:
            base_stride *= max(1, int(gui_playback_keypoint_multiplier.value))
        return min(base_stride, max(1, n_total))

    def rebuild_frame_point_cache() -> None:
        nonlocal frame_point_cache
        frame_point_cache = [get_points_at_time(t) for t in range(max(n_valid, 1))]

    def remove_handle(handle) -> None:
        if handle is None:
            return
        try:
            handle.remove()
        except KeyError:
            pass

    def clear_handle_list(handles: list) -> None:
        for handle in handles:
            remove_handle(handle)
        handles.clear()

    def build_preloaded_keypoint_handles() -> None:
        estimated_primary_points = sum(int(len(points_t)) for points_t, _, _, _ in frame_point_cache)
        estimated_secondary_points = sum(int(len(ghost_points_t)) for _, _, ghost_points_t, _ in frame_point_cache)
        logger.info(
            "开始预加载 keypoint frames: "
            f"frames={len(frame_point_cache)}, primary_points_total={estimated_primary_points}, "
            f"secondary_points_total={estimated_secondary_points}"
        )
        total_primary_points = 0
        total_secondary_points = 0
        with server.atomic():
            clear_handle_list(preloaded_primary_handles)
            clear_handle_list(preloaded_secondary_handles)
            for t, (points_t, colors_t, ghost_points_t, ghost_colors_t) in enumerate(frame_point_cache):
                primary_handle = server.scene.add_point_cloud(
                    name=f"keypoints_primary_preloaded_{t}",
                    points=points_t,
                    colors=colors_to_uint8(colors_t),
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                    precision="float32",
                )
                primary_handle.visible = False
                secondary_handle = server.scene.add_point_cloud(
                    name=f"keypoints_secondary_preloaded_{t}",
                    points=ghost_points_t,
                    colors=colors_to_uint8(ghost_colors_t),
                    point_size=max(0.001, gui_point_size.value * 0.8),
                    point_shape="rounded",
                    precision="float32",
                )
                secondary_handle.visible = False
                preloaded_primary_handles.append(primary_handle)
                preloaded_secondary_handles.append(secondary_handle)
                total_primary_points += int(len(points_t))
                total_secondary_points += int(len(ghost_points_t))
        last_visible_keypoint_frame[0] = -1
        server.flush()
        logger.info(
            "preloaded keypoint frames: "
            f"frames={len(frame_point_cache)}, primary_points_total={total_primary_points}, "
            f"secondary_points_total={total_secondary_points}"
        )

    def build_preloaded_dense_handles() -> None:
        if dense_per_frame is None or dense_colors_per_frame is None:
            return
        estimated_dense_points = sum(int(len(pts)) for pts in dense_per_frame)
        logger.info(
            "开始预加载 dense frames: "
            f"frames={len(dense_per_frame)}, dense_points_total={estimated_dense_points}"
        )
        total_dense_points = 0
        with server.atomic():
            clear_handle_list(preloaded_dense_handles)
            for dense_idx, (pts, cols) in enumerate(zip(dense_per_frame, dense_colors_per_frame)):
                dense_handle = server.scene.add_point_cloud(
                    name=f"dense_pointcloud_preloaded_{dense_idx}",
                    points=pts,
                    colors=cols,
                    point_size=gui_dense_point_size.value if gui_dense_point_size is not None else 0.015,
                    point_shape="rounded",
                )
                dense_handle.visible = False
                preloaded_dense_handles.append(dense_handle)
                total_dense_points += int(len(pts))
        last_visible_dense_handle[0] = -1
        server.flush()
        logger.info(
            "preloaded dense frames: "
            f"frames={len(preloaded_dense_handles)}, dense_points_total={total_dense_points}"
        )

    def rebuild_keypoint_pointcloud() -> None:
        nonlocal primary_point_cloud_handle, secondary_point_cloud_handle
        if use_preload_playback:
            build_preloaded_keypoint_handles()
            update_display(force_dense_exact=True)
            return
        points_t, colors_t, ghost_points_t, ghost_colors_t = frame_point_cache[min(int(gui_time.value), max(n_valid - 1, 0))]
        with server.atomic():
            try:
                primary_point_cloud_handle.remove()
            except KeyError:
                pass
            try:
                secondary_point_cloud_handle.remove()
            except KeyError:
                pass
            primary_point_cloud_handle = server.scene.add_point_cloud(
                name="keypoints_primary",
                points=points_t,
                colors=colors_to_uint8(colors_t),
                point_size=gui_point_size.value,
                point_shape="rounded",
                precision="float32",
            )
            secondary_point_cloud_handle = server.scene.add_point_cloud(
                name="keypoints_secondary",
                points=ghost_points_t,
                colors=colors_to_uint8(ghost_colors_t),
                point_size=max(0.001, gui_point_size.value * 0.8),
                point_shape="rounded",
                precision="float32",
            )
        server.flush()

    def apply_keypoint_stride(*, force: bool = False) -> None:
        nonlocal traj_sub, render_primary_sub, render_secondary_sub, n_show, colors, ghost_colors, motion_order
        current_stride = resolve_active_keypoint_stride()
        if not force and current_stride == active_keypoint_stride[0]:
            return
        current_indices = np.arange(0, n_total, current_stride)
        traj_sub = traj_full[current_indices]
        current_primary_full, current_secondary_full = resolve_render_masks(str(gui_render_mode.value))
        render_primary_sub = current_primary_full[current_indices]
        render_secondary_sub = current_secondary_full[current_indices]
        n_show = len(current_indices)
        active_keypoint_stride[0] = current_stride
        gui_keypoint_count.value = n_show
        color_seed = traj_sub[:, :1, :] if traj_sub.shape[1] > 0 else np.zeros((n_show, 1, 3), dtype=np.float32)
        colors = get_track_colors(color_seed)
        ghost_colors = fade_track_colors(colors)
        motion_order, _ = compute_motion_rank(traj_sub)
        rebuild_frame_point_cache()
        rebuild_keypoint_pointcloud()
        update_display(force_dense_exact=True)
        if gui_show_trails.value:
            update_trails()

    def update_display(*, force_dense_exact: bool = False) -> None:
        t = min(int(gui_time.value), max(n_valid - 1, 0))
        points_t, colors_t, ghost_points_t, ghost_colors_t = frame_point_cache[t]
        with server.atomic():
            if use_preload_playback:
                current_keypoint_visible = gui_show_keypoints.value
                current_secondary_visible = (
                    current_keypoint_visible
                    and str(gui_render_mode.value) == RENDER_MODE_HYBRID
                    and len(ghost_points_t) > 0
                )
                if preloaded_primary_handles:
                    prev_keypoint_frame = last_visible_keypoint_frame[0]
                    if 0 <= prev_keypoint_frame < len(preloaded_primary_handles) and prev_keypoint_frame != t:
                        preloaded_primary_handles[prev_keypoint_frame].visible = False
                        preloaded_secondary_handles[prev_keypoint_frame].visible = False
                    preloaded_primary_handles[t].visible = current_keypoint_visible
                    preloaded_secondary_handles[t].visible = current_secondary_visible
                    last_visible_keypoint_frame[0] = t
                if preloaded_dense_handles and dense_per_frame is not None:
                    dense_idx = resolve_dense_frame_index(t, force_exact=force_dense_exact)
                    prev_dense_idx = last_visible_dense_handle[0]
                    if 0 <= prev_dense_idx < len(preloaded_dense_handles) and prev_dense_idx != dense_idx:
                        preloaded_dense_handles[prev_dense_idx].visible = False
                    preloaded_dense_handles[dense_idx].visible = (
                        gui_show_dense.value if gui_show_dense is not None else True
                    )
                    last_visible_dense_handle[0] = dense_idx
            else:
                primary_point_cloud_handle.points = points_t
                primary_point_cloud_handle.colors = colors_to_uint8(colors_t)
                primary_point_cloud_handle.point_size = gui_point_size.value
                primary_point_cloud_handle.visible = gui_show_keypoints.value
                secondary_point_cloud_handle.points = ghost_points_t
                secondary_point_cloud_handle.colors = colors_to_uint8(ghost_colors_t)
                secondary_point_cloud_handle.point_size = max(0.001, gui_point_size.value * 0.8)
                secondary_point_cloud_handle.visible = (
                    gui_show_keypoints.value
                    and str(gui_render_mode.value) == RENDER_MODE_HYBRID
                    and len(ghost_points_t) > 0
                )
                if dense_point_cloud_handle is not None and dense_per_frame is not None:
                    dense_idx = resolve_dense_frame_index(t, force_exact=force_dense_exact)
                    if dense_idx != last_dense_frame[0]:
                        dense_point_cloud_handle.points = dense_per_frame[dense_idx]
                        dense_point_cloud_handle.colors = dense_colors_per_frame[dense_idx]
                        last_dense_frame[0] = dense_idx
                    if gui_dense_point_size is not None:
                        dense_point_cloud_handle.point_size = gui_dense_point_size.value
                    dense_point_cloud_handle.visible = gui_show_dense.value if gui_show_dense is not None else True
        server.flush()

    def update_trails() -> None:
        primary_segs = []
        primary_cols = []
        secondary_segs = []
        secondary_cols = []
        should_draw = bool(gui_show_trails.value)
        if should_draw:
            t = int(gui_time.value)
            t_end = n_valid - 1 if gui_trail_full.value else t
            t_end = min(t_end, traj_sub.shape[1] - 1)
            if t_end >= 1:
                if gui_trail_dynamic_only.value:
                    n_dynamic = max(10, int(max(n_show, 1) * gui_trail_dynamic_ratio.value))
                    draw_indices = motion_order[:n_dynamic]
                else:
                    draw_indices = np.arange(n_show)

                for i in draw_indices:
                    if i >= traj_sub.shape[0]:
                        continue
                    for j in range(t_end):
                        p0 = traj_sub[i, j, :]
                        p1 = traj_sub[i, j + 1, :]
                        if (
                            render_primary_sub[i, j]
                            and render_primary_sub[i, j + 1]
                            and np.isfinite(p0).all()
                            and np.isfinite(p1).all()
                        ):
                            primary_segs.append([p0, p1])
                            primary_cols.append([colors[i], colors[i]])
                        elif (
                            str(gui_render_mode.value) == RENDER_MODE_HYBRID
                            and render_secondary_sub[i, j]
                            and render_secondary_sub[i, j + 1]
                            and np.isfinite(p0).all()
                            and np.isfinite(p1).all()
                        ):
                            secondary_segs.append([p0, p1])
                            secondary_cols.append([ghost_colors[i], ghost_colors[i]])
            else:
                should_draw = False
        if should_draw and not primary_segs and not secondary_segs:
            should_draw = False

        with server.atomic():
            for handle in trail_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            trail_handles.clear()
            if should_draw:
                trail_name_counter[0] += 1
                if secondary_segs:
                    ghost_trail_handle = server.scene.add_line_segments(
                        name=f"trails_secondary_{trail_name_counter[0]}",
                        points=np.asarray(secondary_segs, dtype=np.float32),
                        colors=np.asarray(secondary_cols, dtype=np.float32),
                        line_width=max(0.5, gui_trail_line_width.value * 0.8),
                    )
                    trail_handles.append(ghost_trail_handle)
                if primary_segs:
                    trail_handle = server.scene.add_line_segments(
                        name=f"trails_primary_{trail_name_counter[0]}",
                        points=np.asarray(primary_segs, dtype=np.float32),
                        colors=np.asarray(primary_cols, dtype=np.float32),
                        line_width=gui_trail_line_width.value,
                    )
                    trail_handles.append(trail_handle)
        server.flush()

    @gui_time.on_update
    def _(_) -> None:
        update_display(force_dense_exact=not gui_playing.value)
        if gui_show_trails.value:
            update_trails()

    @gui_keypoint_stride.on_update
    def _(_) -> None:
        apply_keypoint_stride(force=True)

    @gui_render_mode.on_update
    def _(_) -> None:
        apply_keypoint_stride(force=True)

    @gui_light_playback.on_update
    def _(_) -> None:
        apply_keypoint_stride(force=True)

    @gui_playback_keypoint_multiplier.on_update
    def _(_) -> None:
        apply_keypoint_stride(force=True)

    @gui_point_size.on_update
    def _(_) -> None:
        with server.atomic():
            if use_preload_playback:
                for handle in preloaded_primary_handles:
                    handle.point_size = gui_point_size.value
                for handle in preloaded_secondary_handles:
                    handle.point_size = max(0.001, gui_point_size.value * 0.8)
            else:
                primary_point_cloud_handle.point_size = gui_point_size.value
                secondary_point_cloud_handle.point_size = max(0.001, gui_point_size.value * 0.8)
        server.flush()

    @gui_show_keypoints.on_update
    def _(_) -> None:
        if use_preload_playback:
            update_display(force_dense_exact=not gui_playing.value)
            return
        with server.atomic():
            primary_point_cloud_handle.visible = gui_show_keypoints.value
            secondary_point_cloud_handle.visible = (
                gui_show_keypoints.value
                and str(gui_render_mode.value) == RENDER_MODE_HYBRID
                and len(secondary_point_cloud_handle.points) > 0
            )
        server.flush()

    @gui_show_trails.on_update
    def _(_) -> None:
        update_trails()

    @gui_trail_full.on_update
    def _(_) -> None:
        update_trails()

    @gui_trail_line_width.on_update
    def _(_) -> None:
        update_trails()

    @gui_trail_dynamic_only.on_update
    def _(_) -> None:
        update_trails()

    @gui_trail_dynamic_ratio.on_update
    def _(_) -> None:
        update_trails()

    if gui_dense_point_size is not None:

        @gui_dense_point_size.on_update
        def _(_) -> None:
            with server.atomic():
                if use_preload_playback:
                    for handle in preloaded_dense_handles:
                        handle.point_size = gui_dense_point_size.value
                elif dense_point_cloud_handle is not None:
                    dense_point_cloud_handle.point_size = gui_dense_point_size.value
            server.flush()

    if gui_show_dense is not None:

        @gui_show_dense.on_update
        def _(_) -> None:
            if use_preload_playback:
                update_display(force_dense_exact=not gui_playing.value)
            elif dense_point_cloud_handle is not None:
                dense_point_cloud_handle.visible = gui_show_dense.value
                server.flush()

    if gui_dense_playback_stride is not None:

        @gui_dense_playback_stride.on_update
        def _(_) -> None:
            update_display(force_dense_exact=not gui_playing.value)

    @gui_playing.on_update
    def _(_) -> None:
        apply_keypoint_stride(force=True)
        if not gui_playing.value:
            update_display(force_dense_exact=True)

    rebuild_frame_point_cache()
    if use_preload_playback:
        build_preloaded_dense_handles()
    apply_keypoint_stride(force=True)
    update_display(force_dense_exact=True)

    logger.info(f"Viser 服务器: http://localhost:{args.port}")
    logger.info("使用滑块或勾选「播放」查看 3D keypoint 动画")

    next_frame_deadline = time.perf_counter()
    was_playing = False
    prev_fps = max(1, int(gui_fps.value))
    try:
        while True:
            current_fps = max(1, int(gui_fps.value))
            is_playing = bool(gui_playing.value) and n_valid > 1
            now = time.perf_counter()
            if not is_playing:
                was_playing = False
                prev_fps = current_fps
                next_frame_deadline = now + 1.0 / current_fps
                time.sleep(0.05)
                continue
            if not was_playing or current_fps != prev_fps:
                next_frame_deadline = now
            if now >= next_frame_deadline:
                frame_step = 1
                if gui_light_playback.value:
                    frame_step = max(1, int(gui_playback_frame_step.value))
                gui_time.value = (int(gui_time.value) + frame_step) % n_valid
                next_frame_deadline += 1.0 / current_fps
                if next_frame_deadline < now - 1.0 / current_fps:
                    next_frame_deadline = now + 1.0 / current_fps
            was_playing = True
            prev_fps = current_fps
            sleep_seconds = max(0.001, min(0.05, next_frame_deadline - time.perf_counter()))
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        logger.info("退出")


if __name__ == "__main__":
    main()
