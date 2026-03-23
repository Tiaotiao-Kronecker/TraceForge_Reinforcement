#!/usr/bin/env python3
"""
在同一个 Viser 3D 场景里对比 baseline / variant 两个 sample NPZ 的轨迹集合差异。

- `baseline-only`: 仅 baseline 的 traj_valid_mask 保留
- `overlap`: baseline 和 variant 都保留
- `variant-only`: 仅 variant 的 traj_valid_mask 保留

默认使用 overlap 轨迹的均值位置，便于把集合差异集中显示在一个视图里。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import (
    SceneReader,
    build_pointcloud_from_frame,
    list_sample_query_frames,
    normalize_sample_data,
    traj_uvz_to_world,
)


BASELINE_ONLY_COLOR = np.array([1.0, 0.35, 0.18], dtype=np.float32)
OVERLAP_COLOR = np.array([1.0, 0.86, 0.15], dtype=np.float32)
VARIANT_ONLY_COLOR = np.array([0.15, 0.72, 1.0], dtype=np.float32)
DIFF_LINK_COLOR = np.array([0.95, 0.95, 0.95], dtype=np.float32)

OVERLAP_SOURCE_OPTIONS = ("mean", "baseline", "variant")
DENSE_SOURCE_OPTIONS = ("baseline", "variant")


@dataclass(frozen=True)
class LoadedSampleBundle:
    episode_dir: Path
    sample_path: Path
    query_frame_idx: int
    segment_frame_indices: np.ndarray
    query_w2c: np.ndarray
    traj_world: np.ndarray
    traj_valid_mask: np.ndarray
    raw_track_count: int


@dataclass(frozen=True)
class ComparisonBundle:
    query_frame_idx: int
    frame_count: int
    track_count: int
    segment_frame_indices: np.ndarray
    normalize_w2c: np.ndarray
    baseline_world: np.ndarray
    variant_world: np.ndarray
    overlap_mean_world: np.ndarray
    overlap_baseline_world: np.ndarray
    overlap_variant_world: np.ndarray
    overlap_shared_step_mask: np.ndarray
    baseline_only_mask: np.ndarray
    overlap_mask: np.ndarray
    variant_only_mask: np.ndarray
    baseline_valid_track_count: int
    variant_valid_track_count: int


def normalize_to_first_frame(traj: np.ndarray, extrinsics_first: np.ndarray) -> np.ndarray:
    if len(traj) == 0:
        return traj
    ones = np.ones((*traj.shape[:2], 1), dtype=traj.dtype)
    traj_h = np.concatenate([traj, ones], axis=-1)
    traj_cam = (extrinsics_first @ traj_h.reshape(-1, 4).T).T.reshape(*traj.shape[:2], 4)
    return traj_cam[..., :3]


def compute_motion_order(traj_sub: np.ndarray) -> np.ndarray:
    if len(traj_sub) == 0:
        return np.zeros((0,), dtype=np.int32)
    valid = np.isfinite(traj_sub).all(axis=-1)
    if traj_sub.shape[1] <= 1:
        return np.arange(traj_sub.shape[0], dtype=np.int32)
    delta = np.diff(traj_sub, axis=1)
    valid_pair = valid[:, :-1] & valid[:, 1:]
    delta_norm = np.where(valid_pair, np.linalg.norm(delta, axis=-1), 0.0)
    motion = np.sum(delta_norm, axis=1)
    return np.argsort(-motion).astype(np.int32)


def load_dense_sequence_from_scene(
    scene_reader: SceneReader,
    *,
    frame_indices: np.ndarray,
    downsample: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
    dense_per_frame: list[np.ndarray] = []
    dense_colors_per_frame: list[np.ndarray] = []
    for frame_idx in np.asarray(frame_indices, dtype=np.int32):
        rgb = scene_reader.get_rgb_frame(int(frame_idx))
        depth = scene_reader.get_depth_frame(int(frame_idx))
        dense_points, dense_colors = build_pointcloud_from_frame(
            depth=depth,
            rgb=rgb,
            intrinsics=intrinsics_all[int(frame_idx)],
            w2c=extrinsics_all[int(frame_idx)],
            downsample=downsample,
        )
        dense_per_frame.append(dense_points)
        dense_colors_per_frame.append(dense_colors)
    return dense_per_frame, dense_colors_per_frame


def _derive_render_step_mask(
    sample: dict[str, np.ndarray | bool | int | float | None],
    traj_uvz: np.ndarray,
    traj_valid_mask: np.ndarray,
    *,
    frame_count: int,
) -> np.ndarray:
    traj_supervision_mask = sample.get("traj_supervision_mask")
    if traj_supervision_mask is not None:
        supervision_mask = np.asarray(traj_supervision_mask).astype(bool, copy=False)
        if supervision_mask.shape == (traj_uvz.shape[0], frame_count):
            return supervision_mask

    valid_steps = sample.get("valid_steps")
    if valid_steps is not None:
        valid_steps = np.asarray(valid_steps).astype(bool, copy=False).reshape(-1)[:frame_count]
        if valid_steps.shape == (frame_count,):
            return np.broadcast_to(valid_steps[None, :], traj_uvz.shape[:2]).copy()

    return np.isfinite(traj_uvz).all(axis=-1)


def resolve_common_query_frame(
    baseline_episode_dir: Path,
    variant_episode_dir: Path,
    query_frame: int | None,
) -> int:
    baseline_available = set(list_sample_query_frames(baseline_episode_dir, baseline_episode_dir.name))
    variant_available = set(list_sample_query_frames(variant_episode_dir, variant_episode_dir.name))
    common = sorted(baseline_available & variant_available)
    if not common:
        raise FileNotFoundError(
            "No common sample query frames found between "
            f"{baseline_episode_dir / 'samples'} and {variant_episode_dir / 'samples'}"
        )

    if query_frame is None:
        return int(common[0])
    if query_frame not in baseline_available:
        raise FileNotFoundError(
            f"query_frame={query_frame} not found under {baseline_episode_dir / 'samples'}"
        )
    if query_frame not in variant_available:
        raise FileNotFoundError(
            f"query_frame={query_frame} not found under {variant_episode_dir / 'samples'}"
        )
    return int(query_frame)


def load_sample_bundle(episode_dir: Path, query_frame: int) -> LoadedSampleBundle:
    sample_path = episode_dir / "samples" / f"{episode_dir.name}_{query_frame}.npz"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    sample = normalize_sample_data(sample_path)
    query_frame_idx = int(sample["query_frame_index"])
    segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32)
    traj_uvz = np.asarray(sample["traj_uvz"], dtype=np.float32)
    traj_valid_mask = np.asarray(sample["traj_valid_mask"]).astype(bool, copy=False)

    if sample.get("frame_aligned", False) and len(segment_frame_indices) < traj_uvz.shape[1]:
        traj_uvz = traj_uvz[:, : len(segment_frame_indices)]

    render_step_mask = _derive_render_step_mask(
        sample,
        traj_uvz,
        traj_valid_mask,
        frame_count=traj_uvz.shape[1],
    )
    finite_step_mask = np.isfinite(traj_uvz).all(axis=-1)
    render_step_mask = np.asarray(render_step_mask, dtype=bool) & finite_step_mask

    traj_uvz = np.array(traj_uvz, dtype=np.float32, copy=True)
    traj_uvz[~render_step_mask] = np.nan

    with SceneReader(episode_dir) as scene_reader:
        intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
        query_intrinsics = intrinsics_all[query_frame_idx].astype(np.float32)
        query_w2c = extrinsics_all[query_frame_idx].astype(np.float32)

    traj_world = traj_uvz_to_world(traj_uvz, query_intrinsics, query_w2c)
    return LoadedSampleBundle(
        episode_dir=episode_dir,
        sample_path=sample_path,
        query_frame_idx=query_frame_idx,
        segment_frame_indices=segment_frame_indices,
        query_w2c=query_w2c,
        traj_world=traj_world,
        traj_valid_mask=traj_valid_mask.astype(bool, copy=False),
        raw_track_count=int(traj_uvz.shape[0]),
    )


def build_comparison_bundle(
    baseline_bundle: LoadedSampleBundle,
    variant_bundle: LoadedSampleBundle,
) -> ComparisonBundle:
    if baseline_bundle.query_frame_idx != variant_bundle.query_frame_idx:
        raise ValueError(
            "Query frame mismatch: "
            f"{baseline_bundle.query_frame_idx} vs {variant_bundle.query_frame_idx}"
        )

    track_count = min(
        baseline_bundle.traj_world.shape[0],
        variant_bundle.traj_world.shape[0],
        len(baseline_bundle.traj_valid_mask),
        len(variant_bundle.traj_valid_mask),
    )
    frame_count = min(
        baseline_bundle.traj_world.shape[1],
        variant_bundle.traj_world.shape[1],
        len(baseline_bundle.segment_frame_indices),
        len(variant_bundle.segment_frame_indices),
    )
    if track_count <= 0 or frame_count <= 0:
        raise ValueError("No common tracks/frames available for comparison")

    baseline_world = baseline_bundle.traj_world[:track_count, :frame_count].astype(np.float32, copy=True)
    variant_world = variant_bundle.traj_world[:track_count, :frame_count].astype(np.float32, copy=True)
    baseline_valid_mask = baseline_bundle.traj_valid_mask[:track_count].astype(bool, copy=False)
    variant_valid_mask = variant_bundle.traj_valid_mask[:track_count].astype(bool, copy=False)

    baseline_only_mask = baseline_valid_mask & ~variant_valid_mask
    overlap_mask = baseline_valid_mask & variant_valid_mask
    variant_only_mask = ~baseline_valid_mask & variant_valid_mask

    overlap_shared_step_mask = (
        overlap_mask[:, None]
        & np.isfinite(baseline_world).all(axis=-1)
        & np.isfinite(variant_world).all(axis=-1)
    )
    overlap_baseline_world = np.where(
        overlap_shared_step_mask[..., None],
        baseline_world,
        np.nan,
    ).astype(np.float32)
    overlap_variant_world = np.where(
        overlap_shared_step_mask[..., None],
        variant_world,
        np.nan,
    ).astype(np.float32)
    overlap_mean_world = np.where(
        overlap_shared_step_mask[..., None],
        0.5 * (baseline_world + variant_world),
        np.nan,
    ).astype(np.float32)

    baseline_segment = baseline_bundle.segment_frame_indices[:frame_count]
    variant_segment = variant_bundle.segment_frame_indices[:frame_count]
    if not np.array_equal(baseline_segment, variant_segment):
        logger.warning(
            "Segment frame indices differ on the common prefix; dense playback will follow baseline frames. "
            f"baseline[:{frame_count}]={baseline_segment.tolist()}, "
            f"variant[:{frame_count}]={variant_segment.tolist()}"
        )

    if not np.allclose(baseline_bundle.query_w2c, variant_bundle.query_w2c, atol=1e-5, rtol=1e-5):
        logger.warning("Query-frame extrinsics differ slightly between baseline and variant outputs")

    return ComparisonBundle(
        query_frame_idx=baseline_bundle.query_frame_idx,
        frame_count=frame_count,
        track_count=track_count,
        segment_frame_indices=baseline_segment.astype(np.int32, copy=True),
        normalize_w2c=baseline_bundle.query_w2c.astype(np.float32, copy=True),
        baseline_world=baseline_world,
        variant_world=variant_world,
        overlap_mean_world=overlap_mean_world,
        overlap_baseline_world=overlap_baseline_world,
        overlap_variant_world=overlap_variant_world,
        overlap_shared_step_mask=overlap_shared_step_mask.astype(bool, copy=False),
        baseline_only_mask=baseline_only_mask.astype(bool, copy=False),
        overlap_mask=overlap_mask.astype(bool, copy=False),
        variant_only_mask=variant_only_mask.astype(bool, copy=False),
        baseline_valid_track_count=int(baseline_valid_mask.sum()),
        variant_valid_track_count=int(variant_valid_mask.sum()),
    )


def select_top_motion_indices(traj: np.ndarray, track_mask: np.ndarray, limit: int) -> np.ndarray:
    indices = np.flatnonzero(track_mask)
    if len(indices) == 0 or limit <= 0:
        return np.zeros((0,), dtype=np.int32)
    if len(indices) <= limit:
        return indices.astype(np.int32)
    motion_order = compute_motion_order(traj[indices])
    return indices[motion_order[:limit]].astype(np.int32)


def points_at_time(traj: np.ndarray, indices: np.ndarray, t: int) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = traj[indices, t, :]
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite].astype(np.float32, copy=False)


def build_trail_segments(
    traj: np.ndarray,
    indices: np.ndarray,
    *,
    t_end: int,
) -> np.ndarray:
    if len(indices) == 0 or t_end < 1:
        return np.zeros((0, 2, 3), dtype=np.float32)

    segments: list[list[np.ndarray]] = []
    for idx in indices:
        traj_i = traj[int(idx), : t_end + 1]
        if traj_i.shape[0] <= 1:
            continue
        finite = np.isfinite(traj_i).all(axis=1)
        for step in range(len(traj_i) - 1):
            if finite[step] and finite[step + 1]:
                segments.append([traj_i[step], traj_i[step + 1]])
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)


def build_overlap_diff_segments(
    baseline_world: np.ndarray,
    variant_world: np.ndarray,
    overlap_indices: np.ndarray,
    *,
    t: int,
) -> np.ndarray:
    if len(overlap_indices) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    baseline_pts = baseline_world[overlap_indices, t, :]
    variant_pts = variant_world[overlap_indices, t, :]
    finite = np.isfinite(baseline_pts).all(axis=1) & np.isfinite(variant_pts).all(axis=1)
    if not np.any(finite):
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([baseline_pts[finite], variant_pts[finite]], axis=1).astype(np.float32)


def color_block(color: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.broadcast_to(color[None, :], (count, 3)).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在一个 3D 场景里对比 baseline/variant 轨迹分组")
    parser.add_argument("--baseline_episode_dir", type=Path, required=True)
    parser.add_argument("--variant_episode_dir", type=Path, required=True)
    parser.add_argument("--query_frame", type=int, default=None)
    parser.add_argument("--baseline_label", type=str, default="baseline")
    parser.add_argument("--variant_label", type=str, default="variant")
    parser.add_argument(
        "--overlap_source",
        type=str,
        default="mean",
        choices=OVERLAP_SOURCE_OPTIONS,
        help="overlap 轨迹显示方式：baseline / mean / variant",
    )
    parser.add_argument(
        "--dense_source",
        type=str,
        default="baseline",
        choices=DENSE_SOURCE_OPTIONS,
        help="dense pointcloud 使用哪个 episode_dir 的 scene artifacts",
    )
    parser.add_argument("--dense_pointcloud", action="store_true")
    parser.add_argument("--dense_downsample", type=int, default=4)
    parser.add_argument("--normalize_camera", action="store_true")
    parser.add_argument("--max_tracks_per_group", type=int, default=200)
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_episode_dir = args.baseline_episode_dir.resolve()
    variant_episode_dir = args.variant_episode_dir.resolve()
    if not baseline_episode_dir.is_dir():
        raise FileNotFoundError(f"Baseline episode dir not found: {baseline_episode_dir}")
    if not variant_episode_dir.is_dir():
        raise FileNotFoundError(f"Variant episode dir not found: {variant_episode_dir}")

    query_frame = resolve_common_query_frame(
        baseline_episode_dir,
        variant_episode_dir,
        args.query_frame,
    )
    baseline_bundle = load_sample_bundle(baseline_episode_dir, query_frame)
    variant_bundle = load_sample_bundle(variant_episode_dir, query_frame)
    comparison = build_comparison_bundle(baseline_bundle, variant_bundle)

    dense_per_frame: list[np.ndarray] | None = None
    dense_colors_per_frame: list[np.ndarray] | None = None
    if args.dense_pointcloud:
        dense_episode_dir = (
            baseline_episode_dir if args.dense_source == "baseline" else variant_episode_dir
        )
        dense_frame_indices = (
            comparison.segment_frame_indices
            if args.dense_source == "baseline"
            else variant_bundle.segment_frame_indices[: comparison.frame_count]
        )
        with SceneReader(dense_episode_dir) as scene_reader:
            dense_per_frame, dense_colors_per_frame = load_dense_sequence_from_scene(
                scene_reader,
                frame_indices=dense_frame_indices,
                downsample=args.dense_downsample,
            )

    baseline_world = comparison.baseline_world
    variant_world = comparison.variant_world
    overlap_mean_world = comparison.overlap_mean_world
    overlap_baseline_world = comparison.overlap_baseline_world
    overlap_variant_world = comparison.overlap_variant_world

    if args.normalize_camera:
        baseline_world = normalize_to_first_frame(baseline_world, comparison.normalize_w2c)
        variant_world = normalize_to_first_frame(variant_world, comparison.normalize_w2c)
        overlap_mean_world = normalize_to_first_frame(overlap_mean_world, comparison.normalize_w2c)
        overlap_baseline_world = normalize_to_first_frame(overlap_baseline_world, comparison.normalize_w2c)
        overlap_variant_world = normalize_to_first_frame(overlap_variant_world, comparison.normalize_w2c)
        if dense_per_frame is not None:
            for idx, points in enumerate(dense_per_frame):
                if len(points) == 0:
                    continue
                points_h = np.concatenate(
                    [points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)],
                    axis=1,
                )
                dense_per_frame[idx] = (comparison.normalize_w2c @ points_h.T).T[:, :3].astype(np.float32)

    baseline_only_indices = select_top_motion_indices(
        baseline_world,
        comparison.baseline_only_mask,
        min(args.max_tracks_per_group, int(comparison.baseline_only_mask.sum())),
    )
    overlap_indices = select_top_motion_indices(
        overlap_mean_world,
        comparison.overlap_mask,
        min(args.max_tracks_per_group, int(comparison.overlap_mask.sum())),
    )
    variant_only_indices = select_top_motion_indices(
        variant_world,
        comparison.variant_only_mask,
        min(args.max_tracks_per_group, int(comparison.variant_only_mask.sum())),
    )

    logger.info(
        f"query_frame={comparison.query_frame_idx}, common_tracks={comparison.track_count}, "
        f"common_frames={comparison.frame_count}"
    )
    logger.info(
        f"{args.baseline_label}: valid={comparison.baseline_valid_track_count}, "
        f"{args.variant_label}: valid={comparison.variant_valid_track_count}"
    )
    logger.info(
        f"groups: baseline-only={int(comparison.baseline_only_mask.sum())}, "
        f"overlap={int(comparison.overlap_mask.sum())}, "
        f"variant-only={int(comparison.variant_only_mask.sum())}"
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")

    def current_overlap_world() -> np.ndarray:
        if gui_overlap_source.value == "baseline":
            return overlap_baseline_world
        if gui_overlap_source.value == "variant":
            return overlap_variant_world
        return overlap_mean_world

    def recalc_selected_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            select_top_motion_indices(
                baseline_world,
                comparison.baseline_only_mask,
                int(gui_baseline_limit.value),
            ),
            select_top_motion_indices(
                overlap_mean_world,
                comparison.overlap_mask,
                int(gui_overlap_limit.value),
            ),
            select_top_motion_indices(
                variant_world,
                comparison.variant_only_mask,
                int(gui_variant_limit.value),
            ),
        )

    with server.gui.add_folder("3D 对比动画"):
        gui_time = server.gui.add_slider(
            "时间步",
            min=0,
            max=max(0, comparison.frame_count - 1),
            step=1,
            initial_value=0,
        )
        gui_playing = server.gui.add_checkbox("播放", True)
        gui_fps = server.gui.add_slider("帧率", min=1, max=60, step=1, initial_value=10)
        gui_overlap_source = server.gui.add_dropdown(
            "Overlap source",
            OVERLAP_SOURCE_OPTIONS,
            initial_value=args.overlap_source,
        )
        gui_point_size = server.gui.add_slider(
            "点大小", min=0.001, max=0.2, step=0.001, initial_value=0.03
        )
        gui_show_trails = server.gui.add_checkbox("显示轨迹线", False)
        gui_trail_full = server.gui.add_checkbox("完整轨迹（显示整段 0→末帧）", True)
        gui_trail_line_width = server.gui.add_slider(
            "轨迹线宽", min=0.5, max=15.0, step=0.5, initial_value=4.0
        )
        gui_show_diff_links = server.gui.add_checkbox("显示 overlap 差异连线", False)
        gui_diff_line_width = server.gui.add_slider(
            "差异线宽", min=0.5, max=10.0, step=0.5, initial_value=2.0
        )
        gui_show_baseline_only = server.gui.add_checkbox(f"显示 {args.baseline_label}-only", True)
        gui_show_overlap = server.gui.add_checkbox("显示 overlap", True)
        gui_show_variant_only = server.gui.add_checkbox(f"显示 {args.variant_label}-only", True)
        gui_baseline_total = server.gui.add_number(
            f"{args.baseline_label}-only 总数",
            initial_value=int(comparison.baseline_only_mask.sum()),
            disabled=True,
        )
        gui_overlap_total = server.gui.add_number(
            "overlap 总数",
            initial_value=int(comparison.overlap_mask.sum()),
            disabled=True,
        )
        gui_variant_total = server.gui.add_number(
            f"{args.variant_label}-only 总数",
            initial_value=int(comparison.variant_only_mask.sum()),
            disabled=True,
        )
        baseline_slider_max = max(1, int(comparison.baseline_only_mask.sum()))
        overlap_slider_max = max(1, int(comparison.overlap_mask.sum()))
        variant_slider_max = max(1, int(comparison.variant_only_mask.sum()))
        gui_baseline_limit = server.gui.add_slider(
            f"{args.baseline_label}-only 显示数",
            min=0,
            max=baseline_slider_max,
            step=1,
            initial_value=min(args.max_tracks_per_group, int(comparison.baseline_only_mask.sum())),
        )
        gui_overlap_limit = server.gui.add_slider(
            "overlap 显示数",
            min=0,
            max=overlap_slider_max,
            step=1,
            initial_value=min(args.max_tracks_per_group, int(comparison.overlap_mask.sum())),
        )
        gui_variant_limit = server.gui.add_slider(
            f"{args.variant_label}-only 显示数",
            min=0,
            max=variant_slider_max,
            step=1,
            initial_value=min(args.max_tracks_per_group, int(comparison.variant_only_mask.sum())),
        )
        if dense_per_frame is not None:
            gui_show_dense = server.gui.add_checkbox("显示密集点云", True)
            gui_dense_point_size = server.gui.add_slider(
                "密集点云大小", min=0.001, max=0.1, step=0.001, initial_value=0.015
            )
        else:
            gui_show_dense = None
            gui_dense_point_size = None

    baseline_only_handle = server.scene.add_point_cloud(
        name="baseline_only_points",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        point_size=gui_point_size.value,
        point_shape="rounded",
    )
    overlap_handle = server.scene.add_point_cloud(
        name="overlap_points",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        point_size=gui_point_size.value,
        point_shape="rounded",
    )
    variant_only_handle = server.scene.add_point_cloud(
        name="variant_only_points",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        point_size=gui_point_size.value,
        point_shape="rounded",
    )

    dense_handle = None
    if dense_per_frame is not None:
        dense_handle = server.scene.add_point_cloud(
            name="dense_pointcloud",
            points=dense_per_frame[0],
            colors=dense_colors_per_frame[0].astype(np.float32),
            point_size=gui_dense_point_size.value if gui_dense_point_size is not None else 0.015,
            point_shape="rounded",
        )

    trail_handles: list = []
    diff_link_handle = None
    selected_indices = [baseline_only_indices, overlap_indices, variant_only_indices]

    def remove_trails() -> None:
        for handle in trail_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        trail_handles.clear()

    def update_points() -> None:
        overlap_world = current_overlap_world()
        t = int(gui_time.value)
        current_baseline_only, current_overlap, current_variant_only = selected_indices

        baseline_only_points = points_at_time(baseline_world, current_baseline_only, t)
        overlap_points = points_at_time(overlap_world, current_overlap, t)
        variant_only_points = points_at_time(variant_world, current_variant_only, t)

        baseline_only_handle.points = baseline_only_points
        baseline_only_handle.colors = color_block(BASELINE_ONLY_COLOR, len(baseline_only_points))
        baseline_only_handle.point_size = gui_point_size.value
        baseline_only_handle.visible = gui_show_baseline_only.value

        overlap_handle.points = overlap_points
        overlap_handle.colors = color_block(OVERLAP_COLOR, len(overlap_points))
        overlap_handle.point_size = gui_point_size.value
        overlap_handle.visible = gui_show_overlap.value

        variant_only_handle.points = variant_only_points
        variant_only_handle.colors = color_block(VARIANT_ONLY_COLOR, len(variant_only_points))
        variant_only_handle.point_size = gui_point_size.value
        variant_only_handle.visible = gui_show_variant_only.value

        if dense_handle is not None and dense_per_frame is not None and dense_colors_per_frame is not None:
            dense_idx = min(t, len(dense_per_frame) - 1)
            dense_handle.points = dense_per_frame[dense_idx]
            dense_handle.colors = dense_colors_per_frame[dense_idx].astype(np.float32)
            if gui_dense_point_size is not None:
                dense_handle.point_size = gui_dense_point_size.value
            dense_handle.visible = gui_show_dense.value if gui_show_dense is not None else True

    def update_diff_links() -> None:
        nonlocal diff_link_handle
        if diff_link_handle is not None:
            try:
                diff_link_handle.remove()
            except KeyError:
                pass
            diff_link_handle = None

        if not gui_show_diff_links.value:
            return

        current_overlap = selected_indices[1]
        segments = build_overlap_diff_segments(
            overlap_baseline_world,
            overlap_variant_world,
            current_overlap,
            t=int(gui_time.value),
        )
        if len(segments) == 0:
            return
        diff_link_handle = server.scene.add_line_segments(
            name="overlap_diff_links",
            points=segments,
            colors=np.broadcast_to(DIFF_LINK_COLOR[None, None, :], segments.shape).astype(np.float32),
            line_width=gui_diff_line_width.value,
        )

    def update_trails() -> None:
        remove_trails()
        if not gui_show_trails.value:
            return

        t = int(gui_time.value)
        t_end = comparison.frame_count - 1 if gui_trail_full.value else t
        overlap_world = current_overlap_world()

        groups = [
            ("baseline_only_trails", baseline_world, selected_indices[0], BASELINE_ONLY_COLOR, gui_show_baseline_only.value),
            ("overlap_trails", overlap_world, selected_indices[1], OVERLAP_COLOR, gui_show_overlap.value),
            ("variant_only_trails", variant_world, selected_indices[2], VARIANT_ONLY_COLOR, gui_show_variant_only.value),
        ]
        for name, traj, indices, color, visible in groups:
            if not visible:
                continue
            segments = build_trail_segments(traj, indices, t_end=t_end)
            if len(segments) == 0:
                continue
            colors = np.broadcast_to(color[None, None, :], segments.shape).astype(np.float32)
            handle = server.scene.add_line_segments(
                name=name,
                points=segments,
                colors=colors,
                line_width=gui_trail_line_width.value,
            )
            trail_handles.append(handle)

    def refresh_selection_and_scene() -> None:
        selected_indices[:] = list(recalc_selected_indices())
        update_points()
        update_trails()
        update_diff_links()

    @gui_time.on_update
    def _(_) -> None:
        update_points()
        update_trails()
        update_diff_links()

    @gui_overlap_source.on_update
    def _(_) -> None:
        update_points()
        update_trails()
        update_diff_links()

    @gui_point_size.on_update
    def _(_) -> None:
        update_points()

    @gui_show_trails.on_update
    def _(_) -> None:
        update_trails()

    @gui_trail_full.on_update
    def _(_) -> None:
        update_trails()

    @gui_trail_line_width.on_update
    def _(_) -> None:
        update_trails()

    @gui_show_diff_links.on_update
    def _(_) -> None:
        update_diff_links()

    @gui_diff_line_width.on_update
    def _(_) -> None:
        update_diff_links()

    @gui_show_baseline_only.on_update
    def _(_) -> None:
        update_points()
        update_trails()

    @gui_show_overlap.on_update
    def _(_) -> None:
        update_points()
        update_trails()
        update_diff_links()

    @gui_show_variant_only.on_update
    def _(_) -> None:
        update_points()
        update_trails()

    @gui_baseline_limit.on_update
    def _(_) -> None:
        refresh_selection_and_scene()

    @gui_overlap_limit.on_update
    def _(_) -> None:
        refresh_selection_and_scene()

    @gui_variant_limit.on_update
    def _(_) -> None:
        refresh_selection_and_scene()

    if gui_show_dense is not None:

        @gui_show_dense.on_update
        def _(_) -> None:
            update_points()

    if gui_dense_point_size is not None:

        @gui_dense_point_size.on_update
        def _(_) -> None:
            update_points()

    refresh_selection_and_scene()

    logger.info(f"Viser 服务器: http://localhost:{args.port}")
    logger.info(
        f"颜色说明: {args.baseline_label}-only={BASELINE_ONLY_COLOR.tolist()}, "
        f"overlap={OVERLAP_COLOR.tolist()}, "
        f"{args.variant_label}-only={VARIANT_ONLY_COLOR.tolist()}"
    )

    try:
        while True:
            if gui_playing.value and comparison.frame_count > 1:
                gui_time.value = (int(gui_time.value) + 1) % comparison.frame_count
            time.sleep(1.0 / max(1, gui_fps.value))
    except KeyboardInterrupt:
        logger.info("退出")


if __name__ == "__main__":
    main()
