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
    SceneReader,
    build_pointcloud_from_frame,
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
    query_frame_idx = int(sample["query_frame_index"])
    intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
    query_intrinsics = intrinsics_all[query_frame_idx].astype(np.float32)
    query_w2c = extrinsics_all[query_frame_idx].astype(np.float32)

    traj_uvz = sample["traj_uvz"].astype(np.float32)
    traj_valid_mask = sample["traj_valid_mask"].astype(bool, copy=False)
    keypoints = sample["keypoints"].astype(np.float32)
    segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32)
    traj_supervision_mask = sample.get("traj_supervision_mask")
    if traj_supervision_mask is not None:
        traj_supervision_mask = np.asarray(traj_supervision_mask).astype(bool, copy=False)

    if sample.get("frame_aligned", False) and len(segment_frame_indices) < traj_uvz.shape[1]:
        traj_uvz = traj_uvz[:, : len(segment_frame_indices)]
        if traj_supervision_mask is not None:
            traj_supervision_mask = traj_supervision_mask[:, : len(segment_frame_indices)]

    raw_num_tracks = int(traj_uvz.shape[0])
    traj_uvz = traj_uvz[traj_valid_mask]
    keypoints = keypoints[traj_valid_mask]
    if traj_supervision_mask is not None and len(traj_valid_mask) == traj_supervision_mask.shape[0]:
        traj_supervision_mask = traj_supervision_mask[traj_valid_mask]
    traj_world = traj_uvz_to_world(traj_uvz, query_intrinsics, query_w2c)

    logger.info(
        f"traj_valid_mask 过滤后轨迹数: {traj_world.shape[0]}/{raw_num_tracks}"
    )

    return {
        "traj_world": traj_world,
        "keypoints": keypoints,
        "query_frame_idx": query_frame_idx,
        "query_w2c": query_w2c,
        "segment_frame_indices": segment_frame_indices,
        "traj_supervision_mask": traj_supervision_mask,
        "raw_num_tracks": raw_num_tracks,
        "filtered_num_tracks": int(traj_world.shape[0]),
    }


def main() -> None:
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
            traj_valid_mask = sample["traj_valid_mask"].astype(bool, copy=False)
            if len(traj_valid_mask) == keypoint_traj.shape[0]:
                keypoint_traj = keypoint_traj[traj_valid_mask]
            supervision_full = sample.get("traj_supervision_mask")
            if supervision_full is not None:
                supervision_full = np.asarray(supervision_full).astype(bool, copy=False)
                if len(traj_valid_mask) == supervision_full.shape[0]:
                    supervision_full = supervision_full[traj_valid_mask]
                supervision_full = supervision_full[:, : keypoint_traj.shape[1]]
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
        supervision_full = np.isfinite(traj_full).all(axis=-1)
    stride = max(1, args.keypoint_stride)
    if n_total > 500 and stride > 1:
        stride = 1
        logger.info(f"Keypoints 总数 {n_total} > 500，默认 stride 设为 1 以显示全部")

    if args.normalize_camera:
        traj_full = normalize_to_first_frame(traj_full, query_w2c)
        if dense_per_frame is not None:
            for idx, pts in enumerate(dense_per_frame):
                if len(pts) == 0:
                    continue
                ones = np.ones((len(pts), 1), dtype=np.float32)
                pts_h = np.hstack([pts.astype(np.float32), ones])
                dense_per_frame[idx] = (query_w2c @ pts_h.T).T[:, :3].astype(np.float32)

    indices = np.arange(0, n_total, stride)
    traj_sub = traj_full[indices]
    supervision_sub = supervision_full[indices]
    n_show = len(indices)
    colors = get_track_colors(traj_sub[:, :1, :] if traj_sub.shape[1] > 0 else np.zeros((n_show, 1, 3), dtype=np.float32))
    motion_order, _motion_scores = compute_motion_rank(traj_sub)

    logger.info(
        f"加载 {sample_path.name}: query_frame={query_frame_idx}, "
        f"layout={layout}, keypoints={n_total}->{n_show}, segment_len={n_valid}"
    )
    if dense_per_frame is not None:
        mode = "dynamic" if len(dense_per_frame) > 1 else "static"
        logger.info(
            f"dense pointcloud: {mode}, frames={len(dense_per_frame)}, "
            f"first_frame_points={len(dense_per_frame[0]) if dense_per_frame else 0}"
        )
    if use_legacy_main_dense:
        logger.info("legacy 主 NPZ dense 模式下，轨迹来自主 NPZ 原始帧对齐坐标")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")

    def get_points_at_time(t: int) -> tuple[np.ndarray, np.ndarray]:
        if n_valid <= 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        t = min(max(int(t), 0), n_valid - 1)
        pts = traj_sub[:, t, :].copy()
        frame_mask = np.isfinite(pts).all(axis=1) & supervision_sub[:, t]
        return pts[frame_mask], colors[frame_mask]

    points_0, colors_0 = get_points_at_time(0)
    point_cloud_handle = server.scene.add_point_cloud(
        name="keypoints",
        points=points_0,
        colors=(np.clip(colors_0 * 255.0, 0, 255)).astype(np.uint8),
        point_size=0.03,
        point_shape="rounded",
        precision="float32",
    )

    dense_point_cloud_handle = None
    if dense_per_frame is not None:
        dense_point_cloud_handle = server.scene.add_point_cloud(
            name="dense_pointcloud",
            points=dense_per_frame[0],
            colors=dense_colors_per_frame[0].astype(np.float32),
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

    def rebuild_keypoint_pointcloud() -> None:
        nonlocal point_cloud_handle
        try:
            point_cloud_handle.remove()
        except KeyError:
            pass
        points_t, colors_t = get_points_at_time(int(gui_time.value))
        point_cloud_handle = server.scene.add_point_cloud(
            name="keypoints",
            points=points_t,
            colors=(np.clip(colors_t * 255.0, 0, 255)).astype(np.uint8),
            point_size=gui_point_size.value,
            point_shape="rounded",
            precision="float32",
        )

    def apply_keypoint_stride() -> None:
        nonlocal traj_sub, supervision_sub, n_show, colors, motion_order
        current_stride = max(1, int(gui_keypoint_stride.value))
        current_indices = np.arange(0, n_total, current_stride)
        traj_sub = traj_full[current_indices]
        supervision_sub = supervision_full[current_indices]
        n_show = len(current_indices)
        gui_keypoint_count.value = n_show
        color_seed = traj_sub[:, :1, :] if traj_sub.shape[1] > 0 else np.zeros((n_show, 1, 3), dtype=np.float32)
        colors = get_track_colors(color_seed)
        motion_order, _ = compute_motion_rank(traj_sub)
        rebuild_keypoint_pointcloud()
        update_display()
        update_trails()

    def update_display() -> None:
        t = min(int(gui_time.value), max(n_valid - 1, 0))
        points_t, colors_t = get_points_at_time(t)
        point_cloud_handle.points = points_t
        point_cloud_handle.colors = (np.clip(colors_t * 255.0, 0, 255)).astype(np.uint8)
        point_cloud_handle.point_size = gui_point_size.value
        point_cloud_handle.visible = gui_show_keypoints.value
        if dense_point_cloud_handle is None or dense_per_frame is None:
            return

        dense_idx = 0 if len(dense_per_frame) == 1 else t
        dense_point_cloud_handle.points = dense_per_frame[dense_idx]
        dense_point_cloud_handle.colors = dense_colors_per_frame[dense_idx].astype(np.float32)
        if gui_dense_point_size is not None:
            dense_point_cloud_handle.point_size = gui_dense_point_size.value
        dense_point_cloud_handle.visible = gui_show_dense.value if gui_show_dense is not None else True

    def update_trails() -> None:
        for handle in trail_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        trail_handles.clear()
        if not gui_show_trails.value:
            return

        t = int(gui_time.value)
        t_end = n_valid - 1 if gui_trail_full.value else t
        t_end = min(t_end, traj_sub.shape[1] - 1)
        if t_end < 1:
            return

        if gui_trail_dynamic_only.value:
            n_dynamic = max(10, int(max(n_show, 1) * gui_trail_dynamic_ratio.value))
            draw_indices = motion_order[:n_dynamic]
        else:
            draw_indices = np.arange(n_show)

        all_segs = []
        all_cols = []
        for i in draw_indices:
            if i >= traj_sub.shape[0]:
                continue
            for j in range(t_end):
                p0 = traj_sub[i, j, :]
                p1 = traj_sub[i, j + 1, :]
                if (
                    supervision_sub[i, j]
                    and supervision_sub[i, j + 1]
                    and np.isfinite(p0).all()
                    and np.isfinite(p1).all()
                ):
                    all_segs.append([p0, p1])
                    all_cols.append([colors[i], colors[i]])

        if not all_segs:
            return

        trail_name_counter[0] += 1
        trail_handle = server.scene.add_line_segments(
            name=f"trails_batched_{trail_name_counter[0]}",
            points=np.asarray(all_segs, dtype=np.float32),
            colors=np.asarray(all_cols, dtype=np.float32),
            line_width=gui_trail_line_width.value,
        )
        trail_handles.append(trail_handle)

    @gui_time.on_update
    def _(_) -> None:
        update_display()
        update_trails()

    @gui_keypoint_stride.on_update
    def _(_) -> None:
        apply_keypoint_stride()

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud_handle.point_size = gui_point_size.value

    @gui_show_keypoints.on_update
    def _(_) -> None:
        point_cloud_handle.visible = gui_show_keypoints.value

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
            if dense_point_cloud_handle is not None:
                dense_point_cloud_handle.point_size = gui_dense_point_size.value

    if gui_show_dense is not None:

        @gui_show_dense.on_update
        def _(_) -> None:
            if dense_point_cloud_handle is not None:
                dense_point_cloud_handle.visible = gui_show_dense.value

    update_display()

    logger.info(f"Viser 服务器: http://localhost:{args.port}")
    logger.info("使用滑块或勾选「播放」查看 3D keypoint 动画")

    try:
        while True:
            if gui_playing.value and n_valid > 1:
                gui_time.value = (int(gui_time.value) + 1) % n_valid
            time.sleep(1.0 / max(1, gui_fps.value))
    except KeyboardInterrupt:
        logger.info("退出")


if __name__ == "__main__":
    main()
