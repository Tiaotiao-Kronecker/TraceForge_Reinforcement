#!/usr/bin/env python3
"""
Interactive 4D reconstruction viewer.

This viewer is inspired by D4RT's "Interactive 4D Reconstruction" and is
built on top of existing TraceForge scene/sample artifacts.

It exposes three visualization modes:
- All Pixels Tracking: aggregate tracked world points from all sample NPZs
- Point Cloud: per-frame dense point cloud reconstructed from depth
- Both: show both layers together
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser
import viser.transforms as tf
from loguru import logger
from matplotlib import colormaps

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import (
    RENDER_MODE_FINITE,
    RENDER_MODE_HYBRID,
    RENDER_MODES,
    SceneReader,
    build_pointcloud_from_frame,
    build_sample_visualization_view,
    list_sample_query_frames,
    normalize_sample_data,
    traj_uvz_to_world,
)

TRAIL_MODE_FULL = "Complete"
TRAIL_MODE_PROGRESSIVE = "Progressive"
TRAIL_MODE_WINDOW = "Windowed"
TRAIL_MODES = (TRAIL_MODE_PROGRESSIVE, TRAIL_MODE_FULL, TRAIL_MODE_WINDOW)

COLOR_MODE_VIDEO = "video"
COLOR_MODE_TURBO = "turbo"
COLOR_MODES = (COLOR_MODE_VIDEO, COLOR_MODE_TURBO)

BUFFER_MODE_AUTO = "auto"
BUFFER_MODE_STREAM = "stream"
BUFFER_MODE_BUFFERED = "buffered"
BUFFER_MODES = (BUFFER_MODE_AUTO, BUFFER_MODE_STREAM, BUFFER_MODE_BUFFERED)

DEFAULT_TRACK_POINT_SIZE = 0.012
DEFAULT_SECONDARY_TRACK_SIZE_RATIO = 0.8
DEFAULT_DENSE_POINT_SIZE = 0.016
DEFAULT_TRAIL_FADE_FRAMES = 18
DEFAULT_TRAIL_FADE_FLOOR = 0.06
DEFAULT_TRAIL_FADE_POWER = 1.8
DEFAULT_DISPLAY_DOWNSAMPLE = 1
MAX_DISPLAY_DOWNSAMPLE = 64
AUTO_BUFFER_MAX_FRAMES = 64
DEFAULT_PLAYBACK_FPS = 6
DEFAULT_RGB_PREVIEW_DOWNSAMPLE = 2
DEFAULT_MAX_DENSE_POINTS_PER_FRAME = 30000
DEFAULT_RGB_WHILE_PLAYING = False
DEFAULT_TRACK_SEED_BORDER_MARGIN_PX = 20
DEFAULT_TRACK_MIN_IN_BOUNDS_RATIO = 0.5
DEFAULT_TRACK_MAX_UV_STEP_PX = 96.0
MAX_TRACK_UV_STEP_PX = 1024.0


@dataclass
class FramePointSet:
    points: np.ndarray
    colors: np.ndarray
    seed_border_dist_px: np.ndarray
    in_bounds_ratio: np.ndarray
    max_uv_step_px: np.ndarray


@dataclass
class FrameLineSet:
    segments: np.ndarray
    colors: np.ndarray
    seed_border_dist_px: np.ndarray
    in_bounds_ratio: np.ndarray
    max_uv_step_px: np.ndarray


@dataclass
class AggregatedTracks:
    primary_by_frame: list[FramePointSet]
    secondary_by_frame: list[FramePointSet]
    primary_lines_by_frame: list[FrameLineSet]
    secondary_lines_by_frame: list[FrameLineSet]
    primary_lines_full: FrameLineSet
    secondary_lines_full: FrameLineSet
    primary_line_times_full: np.ndarray
    secondary_line_times_full: np.ndarray
    query_frames: list[int]
    sample_count: int
    total_tracks_after_stride: int
    total_primary_points: int
    total_secondary_points: int
    total_primary_segments: int
    total_secondary_segments: int


@dataclass(frozen=True)
class DisplaySettings:
    track_downsample: int
    dense_downsample: int
    trail_downsample: int
    trail_mode: str
    fade_trails: bool
    fade_frames: int
    trail_window: int
    seed_border_margin_px: int
    min_in_bounds_ratio: float
    max_uv_step_px: float


@dataclass
class DisplayFrameData:
    primary: FramePointSet
    secondary: FramePointSet
    dense: FramePointSet
    primary_lines: FrameLineSet
    secondary_lines: FrameLineSet


@dataclass
class BufferedFrameHandles:
    primary: object
    secondary: object
    dense: object
    primary_lines: object
    secondary_lines: object


def resolve_buffer_mode(raw_mode: str, total_frames: int) -> str:
    raw_mode = str(raw_mode)
    if raw_mode == BUFFER_MODE_AUTO:
        return BUFFER_MODE_BUFFERED if int(total_frames) <= AUTO_BUFFER_MAX_FRAMES else BUFFER_MODE_STREAM
    if raw_mode not in BUFFER_MODES:
        raise ValueError(f"Unsupported buffer mode: {raw_mode}")
    return raw_mode


def prepare_rgb_preview(rgb: np.ndarray, downsample: int) -> np.ndarray:
    downsample = max(int(downsample), 1)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if downsample <= 1:
        return rgb
    return rgb[::downsample, ::downsample]


def empty_frame_line_set() -> FrameLineSet:
    return FrameLineSet(
        segments=np.zeros((0, 2, 3), dtype=np.float32),
        colors=np.zeros((0, 2, 3), dtype=np.uint8),
        seed_border_dist_px=np.zeros((0,), dtype=np.float32),
        in_bounds_ratio=np.zeros((0,), dtype=np.float32),
        max_uv_step_px=np.zeros((0,), dtype=np.float32),
    )


def empty_frame_point_set() -> FramePointSet:
    return FramePointSet(
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        seed_border_dist_px=np.zeros((0,), dtype=np.float32),
        in_bounds_ratio=np.zeros((0,), dtype=np.float32),
        max_uv_step_px=np.zeros((0,), dtype=np.float32),
    )


def downsample_frame_points(point_set: FramePointSet, stride: int) -> FramePointSet:
    stride = max(int(stride), 1)
    if stride <= 1 or len(point_set.points) <= 1:
        return point_set
    return FramePointSet(
        points=point_set.points[::stride].astype(np.float32, copy=False),
        colors=point_set.colors[::stride],
        seed_border_dist_px=point_set.seed_border_dist_px[::stride].astype(np.float32, copy=False),
        in_bounds_ratio=point_set.in_bounds_ratio[::stride].astype(np.float32, copy=False),
        max_uv_step_px=point_set.max_uv_step_px[::stride].astype(np.float32, copy=False),
    )


def downsample_frame_lines(line_set: FrameLineSet, stride: int) -> FrameLineSet:
    stride = max(int(stride), 1)
    if stride <= 1 or len(line_set.segments) <= 1:
        return line_set
    return FrameLineSet(
        segments=line_set.segments[::stride].astype(np.float32, copy=False),
        colors=line_set.colors[::stride],
        seed_border_dist_px=line_set.seed_border_dist_px[::stride].astype(np.float32, copy=False),
        in_bounds_ratio=line_set.in_bounds_ratio[::stride].astype(np.float32, copy=False),
        max_uv_step_px=line_set.max_uv_step_px[::stride].astype(np.float32, copy=False),
    )


def get_track_colors(seed_points: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    if len(seed_points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(seed_points, dtype=np.float32).reshape(len(seed_points), -1)
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return np.full((len(seed_points), 3), 0.5, dtype=np.float32)
    valid_pts = pts[finite]
    mins = valid_pts.min(axis=0)
    maxs = valid_pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = np.clip((pts - mins) / span, 0.0, 1.0)
    score = normalized.sum(axis=1)
    denom = max(len(score) - 1, 1)
    rank = np.argsort(np.argsort(score)).astype(np.float32) / float(denom)
    cmap = colormaps[colormap]
    return np.asarray([cmap(float(v))[:3] for v in rank], dtype=np.float32)


def fade_colors(colors: np.ndarray, blend: float = 0.72) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    blend = float(np.clip(blend, 0.0, 1.0))
    return colors * (1.0 - blend) + blend


def sample_rgb_at_uv(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float32)
    if uv.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    colors = np.full((len(uv), 3), 127, dtype=np.uint8)
    if np.any(in_bounds):
        colors[in_bounds] = np.asarray(rgb, dtype=np.uint8)[v[in_bounds], u[in_bounds], :3]
    return colors


def deterministic_limit(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)
    max_points = max(int(max_points), 1)
    if len(points) <= max_points:
        return points, colors
    indices = deterministic_sample_indices(len(points), max_points)
    return points[indices], colors[indices]


def deterministic_sample_indices(length: int, max_items: int) -> np.ndarray:
    length = int(length)
    max_items = max(int(max_items), 1)
    if length <= max_items:
        return np.arange(length, dtype=np.int32)
    stride = int(math.ceil(length / float(max_items)))
    return np.arange(0, length, stride, dtype=np.int32)[:max_items]


def concat_frame_chunks(
    point_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    seed_border_chunks: list[np.ndarray],
    in_bounds_ratio_chunks: list[np.ndarray],
    max_uv_step_chunks: list[np.ndarray],
    max_points: int,
) -> FramePointSet:
    if not point_chunks:
        return empty_frame_point_set()
    points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(color_chunks, axis=0)
    seed_border_dist_px = np.concatenate(seed_border_chunks, axis=0).astype(np.float32, copy=False)
    in_bounds_ratio = np.concatenate(in_bounds_ratio_chunks, axis=0).astype(np.float32, copy=False)
    max_uv_step_px = np.concatenate(max_uv_step_chunks, axis=0).astype(np.float32, copy=False)
    indices = deterministic_sample_indices(len(points), max_points)
    return FramePointSet(
        points=points[indices],
        colors=colors[indices],
        seed_border_dist_px=seed_border_dist_px[indices],
        in_bounds_ratio=in_bounds_ratio[indices],
        max_uv_step_px=max_uv_step_px[indices],
    )


def concat_frame_line_chunks(
    segment_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    seed_border_chunks: list[np.ndarray],
    in_bounds_ratio_chunks: list[np.ndarray],
    max_uv_step_chunks: list[np.ndarray],
    max_segments: int,
) -> FrameLineSet:
    if not segment_chunks:
        return empty_frame_line_set()
    segments = np.concatenate(segment_chunks, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(color_chunks, axis=0)
    seed_border_dist_px = np.concatenate(seed_border_chunks, axis=0).astype(np.float32, copy=False)
    in_bounds_ratio = np.concatenate(in_bounds_ratio_chunks, axis=0).astype(np.float32, copy=False)
    max_uv_step_px = np.concatenate(max_uv_step_chunks, axis=0).astype(np.float32, copy=False)
    indices = deterministic_sample_indices(len(segments), max_segments)
    segments = segments[indices]
    colors = colors[indices]
    return FrameLineSet(
        segments=segments,
        colors=colors,
        seed_border_dist_px=seed_border_dist_px[indices],
        in_bounds_ratio=in_bounds_ratio[indices],
        max_uv_step_px=max_uv_step_px[indices],
    )


def concat_timed_line_chunks(
    segment_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    time_chunks: list[np.ndarray],
    seed_border_chunks: list[np.ndarray],
    in_bounds_ratio_chunks: list[np.ndarray],
    max_uv_step_chunks: list[np.ndarray],
    max_segments: int,
) -> tuple[FrameLineSet, np.ndarray]:
    if not segment_chunks:
        return (
            empty_frame_line_set(),
            np.zeros((0,), dtype=np.int32),
        )
    segments = np.concatenate(segment_chunks, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(color_chunks, axis=0)
    times = np.concatenate(time_chunks, axis=0).astype(np.int32, copy=False)
    seed_border_dist_px = np.concatenate(seed_border_chunks, axis=0).astype(np.float32, copy=False)
    in_bounds_ratio = np.concatenate(in_bounds_ratio_chunks, axis=0).astype(np.float32, copy=False)
    max_uv_step_px = np.concatenate(max_uv_step_chunks, axis=0).astype(np.float32, copy=False)
    indices = deterministic_sample_indices(len(segments), max_segments)
    return (
        FrameLineSet(
            segments=segments[indices],
            colors=colors[indices],
            seed_border_dist_px=seed_border_dist_px[indices],
            in_bounds_ratio=in_bounds_ratio[indices],
            max_uv_step_px=max_uv_step_px[indices],
        ),
        times[indices],
    )


def compute_track_filter_metrics(
    *,
    keypoints: np.ndarray,
    traj_uvz_finite: np.ndarray,
    traj_2d_finite: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    traj_uvz_finite = np.asarray(traj_uvz_finite, dtype=np.float32)
    traj_2d_finite = np.asarray(traj_2d_finite, dtype=np.float32)
    if keypoints.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty

    border_dist = np.minimum.reduce(
        [
            keypoints[:, 0],
            keypoints[:, 1],
            np.maximum(float(image_width - 1) - keypoints[:, 0], 0.0),
            np.maximum(float(image_height - 1) - keypoints[:, 1], 0.0),
        ]
    ).astype(np.float32, copy=False)

    finite_uv = np.isfinite(traj_2d_finite).all(axis=-1)
    valid_depth = np.isfinite(traj_uvz_finite).all(axis=-1)
    valid_depth &= traj_uvz_finite[..., 2] > 0.01
    valid_depth &= traj_uvz_finite[..., 2] < 50.0
    in_bounds = valid_depth.copy()
    in_bounds &= traj_2d_finite[..., 0] >= 0.0
    in_bounds &= traj_2d_finite[..., 0] <= float(max(image_width - 1, 0))
    in_bounds &= traj_2d_finite[..., 1] >= 0.0
    in_bounds &= traj_2d_finite[..., 1] <= float(max(image_height - 1, 0))

    valid_counts = valid_depth.sum(axis=1).astype(np.float32, copy=False)
    in_bounds_ratio = np.zeros(keypoints.shape[0], dtype=np.float32)
    has_valid_depth = valid_counts > 0
    if np.any(has_valid_depth):
        in_bounds_ratio[has_valid_depth] = (
            in_bounds.sum(axis=1)[has_valid_depth].astype(np.float32, copy=False)
            / valid_counts[has_valid_depth]
        )

    max_uv_step_px = np.zeros(keypoints.shape[0], dtype=np.float32)
    if traj_2d_finite.shape[1] > 1:
        uv_step = np.linalg.norm(np.diff(traj_2d_finite, axis=1), axis=-1).astype(np.float32, copy=False)
        uv_step_valid = finite_uv[:, 1:] & finite_uv[:, :-1]
        uv_step[~uv_step_valid] = np.nan
        has_uv_step = np.any(uv_step_valid, axis=1)
        if np.any(has_uv_step):
            with np.errstate(all="ignore"):
                uv_step_max = np.nanmax(uv_step, axis=1).astype(np.float32, copy=False)
            max_uv_step_px[has_uv_step] = np.nan_to_num(
                uv_step_max[has_uv_step],
                nan=0.0,
                posinf=MAX_TRACK_UV_STEP_PX * 10.0,
                neginf=0.0,
            )

    return (
        border_dist.astype(np.float32, copy=False),
        in_bounds_ratio.astype(np.float32, copy=False),
        max_uv_step_px.astype(np.float32, copy=False),
    )


def build_track_display_mask(
    seed_border_dist_px: np.ndarray,
    in_bounds_ratio: np.ndarray,
    max_uv_step_px: np.ndarray,
    settings: DisplaySettings,
) -> np.ndarray:
    keep = np.ones(len(seed_border_dist_px), dtype=bool)
    if len(keep) == 0:
        return keep
    if int(settings.seed_border_margin_px) > 0:
        keep &= np.asarray(seed_border_dist_px, dtype=np.float32) >= float(settings.seed_border_margin_px)
    if float(settings.min_in_bounds_ratio) > 0.0:
        keep &= np.asarray(in_bounds_ratio, dtype=np.float32) >= float(settings.min_in_bounds_ratio)
    if float(settings.max_uv_step_px) > 0.0:
        keep &= np.asarray(max_uv_step_px, dtype=np.float32) <= float(settings.max_uv_step_px)
    return keep


def filter_frame_points(point_set: FramePointSet, settings: DisplaySettings) -> FramePointSet:
    keep = build_track_display_mask(
        point_set.seed_border_dist_px,
        point_set.in_bounds_ratio,
        point_set.max_uv_step_px,
        settings,
    )
    if keep.size == 0 or np.all(keep):
        return point_set
    if not np.any(keep):
        return empty_frame_point_set()
    return FramePointSet(
        points=point_set.points[keep].astype(np.float32, copy=False),
        colors=point_set.colors[keep],
        seed_border_dist_px=point_set.seed_border_dist_px[keep].astype(np.float32, copy=False),
        in_bounds_ratio=point_set.in_bounds_ratio[keep].astype(np.float32, copy=False),
        max_uv_step_px=point_set.max_uv_step_px[keep].astype(np.float32, copy=False),
    )


def filter_frame_lines(line_set: FrameLineSet, settings: DisplaySettings) -> FrameLineSet:
    keep = build_track_display_mask(
        line_set.seed_border_dist_px,
        line_set.in_bounds_ratio,
        line_set.max_uv_step_px,
        settings,
    )
    if keep.size == 0 or np.all(keep):
        return line_set
    if not np.any(keep):
        return empty_frame_line_set()
    return FrameLineSet(
        segments=line_set.segments[keep].astype(np.float32, copy=False),
        colors=line_set.colors[keep],
        seed_border_dist_px=line_set.seed_border_dist_px[keep].astype(np.float32, copy=False),
        in_bounds_ratio=line_set.in_bounds_ratio[keep].astype(np.float32, copy=False),
        max_uv_step_px=line_set.max_uv_step_px[keep].astype(np.float32, copy=False),
    )


def compute_trail_fade_weights(
    ages: np.ndarray,
    *,
    fade_frames: int,
    min_brightness: float,
    power: float,
) -> np.ndarray:
    ages = np.asarray(ages, dtype=np.float32)
    fade_frames = max(int(fade_frames), 1)
    min_brightness = float(np.clip(min_brightness, 0.0, 1.0))
    power = max(float(power), 1e-3)
    normalized = 1.0 - np.clip(ages / float(fade_frames), 0.0, 1.0)
    return min_brightness + (1.0 - min_brightness) * np.power(normalized, power)


def build_time_faded_line_set(
    full_lines: FrameLineSet,
    line_times: np.ndarray,
    *,
    frame_idx: int,
    start_idx: int,
    fade_frames: int,
    enable_fade: bool,
) -> FrameLineSet:
    if len(full_lines.segments) == 0:
        return empty_frame_line_set()

    line_times = np.asarray(line_times, dtype=np.int32).reshape(-1)
    if line_times.size == 0:
        return empty_frame_line_set()

    frame_idx = int(frame_idx)
    start_idx = int(start_idx)
    fade_frames = max(int(fade_frames), 1)
    ages_end = frame_idx - line_times
    valid = (line_times >= start_idx) & (line_times <= frame_idx)
    if enable_fade:
        valid &= ages_end < fade_frames
    if not np.any(valid):
        return empty_frame_line_set()

    segments = full_lines.segments[valid]
    colors = np.asarray(full_lines.colors[valid], dtype=np.float32)
    seed_border_dist_px = full_lines.seed_border_dist_px[valid].astype(np.float32, copy=False)
    in_bounds_ratio = full_lines.in_bounds_ratio[valid].astype(np.float32, copy=False)
    max_uv_step_px = full_lines.max_uv_step_px[valid].astype(np.float32, copy=False)
    if not enable_fade:
        return FrameLineSet(
            segments=segments.astype(np.float32, copy=False),
            colors=np.clip(colors, 0.0, 255.0).astype(np.uint8),
            seed_border_dist_px=seed_border_dist_px,
            in_bounds_ratio=in_bounds_ratio,
            max_uv_step_px=max_uv_step_px,
        )

    ages_end = ages_end[valid].astype(np.float32, copy=False)
    endpoint_ages = np.stack([ages_end + 1.0, ages_end], axis=1)
    endpoint_weights = compute_trail_fade_weights(
        endpoint_ages,
        fade_frames=fade_frames,
        min_brightness=DEFAULT_TRAIL_FADE_FLOOR,
        power=DEFAULT_TRAIL_FADE_POWER,
    )
    faded_colors = np.clip(colors * endpoint_weights[..., None], 0.0, 255.0).astype(np.uint8)
    return FrameLineSet(
        segments=segments.astype(np.float32, copy=False),
        colors=faded_colors,
        seed_border_dist_px=seed_border_dist_px,
        in_bounds_ratio=in_bounds_ratio,
        max_uv_step_px=max_uv_step_px,
    )


def build_dense_pointcloud_for_frame(
    scene_reader: SceneReader,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    frame_idx: int,
    downsample: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth = scene_reader.get_depth_frame(int(frame_idx))
    rgb = scene_reader.get_rgb_frame(int(frame_idx))
    points, colors = build_pointcloud_from_frame(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics[int(frame_idx)],
        w2c=extrinsics[int(frame_idx)],
        downsample=max(1, int(downsample)),
    )
    points = points.astype(np.float32)
    colors = np.asarray(colors)
    if int(max_points) > 0:
        points, colors = deterministic_limit(points, colors, max_points=int(max_points))
    return points, colors


def build_display_lines_for_frame(
    *,
    aggregated: AggregatedTracks,
    frame_idx: int,
    settings: DisplaySettings,
) -> tuple[FrameLineSet, FrameLineSet]:
    trail_mode = str(settings.trail_mode)
    if trail_mode == TRAIL_MODE_FULL:
        primary_lines = aggregated.primary_lines_full
        secondary_lines = aggregated.secondary_lines_full
    elif trail_mode == TRAIL_MODE_PROGRESSIVE:
        primary_lines = build_time_faded_line_set(
            aggregated.primary_lines_full,
            aggregated.primary_line_times_full,
            frame_idx=int(frame_idx),
            start_idx=1,
            fade_frames=int(settings.fade_frames),
            enable_fade=bool(settings.fade_trails),
        )
        secondary_lines = build_time_faded_line_set(
            aggregated.secondary_lines_full,
            aggregated.secondary_line_times_full,
            frame_idx=int(frame_idx),
            start_idx=1,
            fade_frames=int(settings.fade_frames),
            enable_fade=bool(settings.fade_trails),
        )
    else:
        start_idx = max(1, int(frame_idx) - max(int(settings.trail_window), 1) + 1)
        effective_fade_frames = min(int(settings.fade_frames), max(int(frame_idx) - start_idx + 1, 1))
        primary_lines = build_time_faded_line_set(
            aggregated.primary_lines_full,
            aggregated.primary_line_times_full,
            frame_idx=int(frame_idx),
            start_idx=start_idx,
            fade_frames=effective_fade_frames,
            enable_fade=bool(settings.fade_trails),
        )
        secondary_lines = build_time_faded_line_set(
            aggregated.secondary_lines_full,
            aggregated.secondary_line_times_full,
            frame_idx=int(frame_idx),
            start_idx=start_idx,
            fade_frames=effective_fade_frames,
            enable_fade=bool(settings.fade_trails),
        )
    primary_lines = filter_frame_lines(primary_lines, settings)
    secondary_lines = filter_frame_lines(secondary_lines, settings)
    primary_lines = downsample_frame_lines(primary_lines, int(settings.trail_downsample))
    secondary_lines = downsample_frame_lines(secondary_lines, int(settings.trail_downsample))
    return primary_lines, secondary_lines


def build_display_frame_data(
    *,
    frame_idx: int,
    aggregated: AggregatedTracks,
    get_dense,
    settings: DisplaySettings,
) -> DisplayFrameData:
    primary = filter_frame_points(aggregated.primary_by_frame[int(frame_idx)], settings)
    secondary = filter_frame_points(aggregated.secondary_by_frame[int(frame_idx)], settings)
    primary = downsample_frame_points(primary, int(settings.track_downsample))
    secondary = downsample_frame_points(secondary, int(settings.track_downsample))
    primary_lines, secondary_lines = build_display_lines_for_frame(
        aggregated=aggregated,
        frame_idx=int(frame_idx),
        settings=settings,
    )
    dense_points, dense_colors = get_dense(int(frame_idx))
    dense = downsample_frame_points(
        FramePointSet(
            points=np.asarray(dense_points, dtype=np.float32),
            colors=np.asarray(dense_colors),
            seed_border_dist_px=np.zeros((len(dense_points),), dtype=np.float32),
            in_bounds_ratio=np.ones((len(dense_points),), dtype=np.float32),
            max_uv_step_px=np.zeros((len(dense_points),), dtype=np.float32),
        ),
        int(settings.dense_downsample),
    )
    return DisplayFrameData(
        primary=primary,
        secondary=secondary,
        dense=dense,
        primary_lines=primary_lines,
        secondary_lines=secondary_lines,
    )


def aggregate_tracked_points(
    *,
    episode_dir: Path,
    scene_reader: SceneReader,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    render_mode: str,
    query_stride: int,
    track_stride: int,
    max_points_per_frame: int,
    max_segments_per_frame: int,
    max_full_segments: int,
    color_mode: str,
) -> AggregatedTracks:
    video_name = episode_dir.name
    query_frames_all = list_sample_query_frames(episode_dir, video_name)
    if not query_frames_all:
        raise FileNotFoundError(f"No sample NPZ files found under {episode_dir / 'samples'}")

    query_stride = max(1, int(query_stride))
    track_stride = max(1, int(track_stride))
    selected_query_frames = query_frames_all[::query_stride]
    total_frames = int(len(intrinsics))
    primary_points: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_colors: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_seed_border_dist: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_in_bounds_ratio: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_max_uv_step: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_points: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_colors: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_seed_border_dist: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_in_bounds_ratio: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_max_uv_step: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_segments: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_segment_colors: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_segment_seed_border_dist: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_segment_in_bounds_ratio: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_segment_max_uv_step: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_segments: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_segment_colors: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_segment_seed_border_dist: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_segment_in_bounds_ratio: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_segment_max_uv_step: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    primary_segment_times: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    secondary_segment_times: list[list[np.ndarray]] = [[] for _ in range(total_frames)]
    rgb_cache: dict[int, np.ndarray] = {}
    total_tracks_after_stride = 0

    def get_rgb(frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx not in rgb_cache:
            rgb_cache[frame_idx] = scene_reader.get_rgb_frame(frame_idx)
        return rgb_cache[frame_idx]

    for query_frame in selected_query_frames:
        sample_path = episode_dir / "samples" / f"{video_name}_{query_frame}.npz"
        sample = normalize_sample_data(sample_path)
        render_view = build_sample_visualization_view(sample, render_mode=render_mode)
        segment_frame_indices = np.asarray(render_view["segment_frame_indices"], dtype=np.int32)
        query_frame_idx = int(sample["query_frame_index"])

        primary_uvz = np.asarray(render_view["traj_uvz"], dtype=np.float32)[::track_stride]
        primary_uv = np.asarray(render_view["traj_2d"], dtype=np.float32)[::track_stride]
        secondary_uvz = np.asarray(render_view["traj_uvz_secondary"], dtype=np.float32)[::track_stride]
        secondary_uv = np.asarray(render_view["traj_2d_secondary"], dtype=np.float32)[::track_stride]
        query_rgb = get_rgb(query_frame_idx)
        track_seed_border_dist, track_in_bounds_ratio, track_max_uv_step = compute_track_filter_metrics(
            keypoints=np.asarray(render_view["keypoints"], dtype=np.float32)[::track_stride],
            traj_uvz_finite=np.asarray(render_view["traj_uvz_finite"], dtype=np.float32)[::track_stride],
            traj_2d_finite=np.asarray(render_view["traj_2d_finite"], dtype=np.float32)[::track_stride],
            image_height=int(query_rgb.shape[0]),
            image_width=int(query_rgb.shape[1]),
        )

        primary_world = traj_uvz_to_world(
            primary_uvz,
            intrinsics[query_frame_idx].astype(np.float32),
            extrinsics[query_frame_idx].astype(np.float32),
        )
        secondary_world = traj_uvz_to_world(
            secondary_uvz,
            intrinsics[query_frame_idx].astype(np.float32),
            extrinsics[query_frame_idx].astype(np.float32),
        )

        total_tracks_after_stride += int(primary_world.shape[0])
        if color_mode == COLOR_MODE_TURBO:
            primary_track_colors = (np.clip(get_track_colors(primary_world[:, :1, :]) * 255.0, 0, 255)).astype(
                np.uint8
            )
            secondary_track_colors = (
                np.clip(fade_colors(get_track_colors(secondary_world[:, :1, :])) * 255.0, 0, 255)
            ).astype(np.uint8)
        else:
            primary_track_colors = None
            secondary_track_colors = None

        for local_t, frame_idx in enumerate(segment_frame_indices.tolist()):
            if frame_idx < 0 or frame_idx >= total_frames:
                continue

            rgb = get_rgb(frame_idx)

            pts_primary = primary_world[:, local_t, :]
            uv_primary = primary_uv[:, local_t, :]
            valid_primary = np.isfinite(pts_primary).all(axis=1) & np.isfinite(uv_primary).all(axis=1)
            if np.any(valid_primary):
                primary_points[frame_idx].append(pts_primary[valid_primary].astype(np.float32, copy=False))
                if primary_track_colors is not None:
                    primary_colors[frame_idx].append(primary_track_colors[valid_primary])
                else:
                    primary_colors[frame_idx].append(sample_rgb_at_uv(rgb, uv_primary[valid_primary]))
                primary_seed_border_dist[frame_idx].append(
                    track_seed_border_dist[valid_primary].astype(np.float32, copy=False)
                )
                primary_in_bounds_ratio[frame_idx].append(
                    track_in_bounds_ratio[valid_primary].astype(np.float32, copy=False)
                )
                primary_max_uv_step[frame_idx].append(
                    track_max_uv_step[valid_primary].astype(np.float32, copy=False)
                )
            if local_t > 0:
                prev_primary = primary_world[:, local_t - 1, :]
                valid_primary_seg = np.isfinite(prev_primary).all(axis=1) & np.isfinite(pts_primary).all(axis=1)
                if np.any(valid_primary_seg):
                    seg = np.stack([prev_primary[valid_primary_seg], pts_primary[valid_primary_seg]], axis=1)
                    primary_segments[frame_idx].append(seg.astype(np.float32, copy=False))
                    if primary_track_colors is not None:
                        seg_cols = np.repeat(primary_track_colors[valid_primary_seg][:, None, :], 2, axis=1)
                    else:
                        seg_cols_single = sample_rgb_at_uv(rgb, uv_primary[valid_primary_seg])
                        seg_cols = np.repeat(seg_cols_single[:, None, :], 2, axis=1)
                    primary_segment_colors[frame_idx].append(seg_cols.astype(np.uint8, copy=False))
                    primary_segment_seed_border_dist[frame_idx].append(
                        track_seed_border_dist[valid_primary_seg].astype(np.float32, copy=False)
                    )
                    primary_segment_in_bounds_ratio[frame_idx].append(
                        track_in_bounds_ratio[valid_primary_seg].astype(np.float32, copy=False)
                    )
                    primary_segment_max_uv_step[frame_idx].append(
                        track_max_uv_step[valid_primary_seg].astype(np.float32, copy=False)
                    )
                    primary_segment_times[frame_idx].append(
                        np.full((len(seg),), int(frame_idx), dtype=np.int32)
                    )

            pts_secondary = secondary_world[:, local_t, :]
            uv_secondary = secondary_uv[:, local_t, :]
            valid_secondary = np.isfinite(pts_secondary).all(axis=1) & np.isfinite(uv_secondary).all(axis=1)
            if np.any(valid_secondary):
                secondary_points[frame_idx].append(pts_secondary[valid_secondary].astype(np.float32, copy=False))
                if secondary_track_colors is not None:
                    secondary_colors[frame_idx].append(secondary_track_colors[valid_secondary])
                else:
                    sampled = sample_rgb_at_uv(rgb, uv_secondary[valid_secondary]).astype(np.float32) / 255.0
                    secondary_colors[frame_idx].append(
                        (np.clip(fade_colors(sampled) * 255.0, 0, 255)).astype(np.uint8)
                    )
                secondary_seed_border_dist[frame_idx].append(
                    track_seed_border_dist[valid_secondary].astype(np.float32, copy=False)
                )
                secondary_in_bounds_ratio[frame_idx].append(
                    track_in_bounds_ratio[valid_secondary].astype(np.float32, copy=False)
                )
                secondary_max_uv_step[frame_idx].append(
                    track_max_uv_step[valid_secondary].astype(np.float32, copy=False)
                )
            if local_t > 0:
                prev_secondary = secondary_world[:, local_t - 1, :]
                valid_secondary_seg = np.isfinite(prev_secondary).all(axis=1) & np.isfinite(pts_secondary).all(axis=1)
                if np.any(valid_secondary_seg):
                    seg = np.stack([prev_secondary[valid_secondary_seg], pts_secondary[valid_secondary_seg]], axis=1)
                    secondary_segments[frame_idx].append(seg.astype(np.float32, copy=False))
                    if secondary_track_colors is not None:
                        seg_cols = np.repeat(secondary_track_colors[valid_secondary_seg][:, None, :], 2, axis=1)
                    else:
                        seg_cols_single = sample_rgb_at_uv(rgb, uv_secondary[valid_secondary_seg]).astype(np.float32) / 255.0
                        seg_cols = np.repeat(
                            (np.clip(fade_colors(seg_cols_single) * 255.0, 0, 255)).astype(np.uint8)[:, None, :],
                            2,
                            axis=1,
                        )
                    secondary_segment_colors[frame_idx].append(seg_cols.astype(np.uint8, copy=False))
                    secondary_segment_seed_border_dist[frame_idx].append(
                        track_seed_border_dist[valid_secondary_seg].astype(np.float32, copy=False)
                    )
                    secondary_segment_in_bounds_ratio[frame_idx].append(
                        track_in_bounds_ratio[valid_secondary_seg].astype(np.float32, copy=False)
                    )
                    secondary_segment_max_uv_step[frame_idx].append(
                        track_max_uv_step[valid_secondary_seg].astype(np.float32, copy=False)
                    )
                    secondary_segment_times[frame_idx].append(
                        np.full((len(seg),), int(frame_idx), dtype=np.int32)
                    )

    primary_by_frame = [
        concat_frame_chunks(
            primary_points[idx],
            primary_colors[idx],
            primary_seed_border_dist[idx],
            primary_in_bounds_ratio[idx],
            primary_max_uv_step[idx],
            max_points=max_points_per_frame,
        )
        for idx in range(total_frames)
    ]
    secondary_by_frame = [
        concat_frame_chunks(
            secondary_points[idx],
            secondary_colors[idx],
            secondary_seed_border_dist[idx],
            secondary_in_bounds_ratio[idx],
            secondary_max_uv_step[idx],
            max_points=max_points_per_frame,
        )
        for idx in range(total_frames)
    ]
    primary_lines_by_frame: list[FrameLineSet] = []
    primary_line_times_by_frame: list[np.ndarray] = []
    secondary_lines_by_frame: list[FrameLineSet] = []
    secondary_line_times_by_frame: list[np.ndarray] = []
    for idx in range(total_frames):
        primary_line_set, primary_times = concat_timed_line_chunks(
            primary_segments[idx],
            primary_segment_colors[idx],
            primary_segment_times[idx],
            primary_segment_seed_border_dist[idx],
            primary_segment_in_bounds_ratio[idx],
            primary_segment_max_uv_step[idx],
            max_segments=max_segments_per_frame,
        )
        secondary_line_set, secondary_times = concat_timed_line_chunks(
            secondary_segments[idx],
            secondary_segment_colors[idx],
            secondary_segment_times[idx],
            secondary_segment_seed_border_dist[idx],
            secondary_segment_in_bounds_ratio[idx],
            secondary_segment_max_uv_step[idx],
            max_segments=max_segments_per_frame,
        )
        primary_lines_by_frame.append(primary_line_set)
        primary_line_times_by_frame.append(primary_times)
        secondary_lines_by_frame.append(secondary_line_set)
        secondary_line_times_by_frame.append(secondary_times)
    primary_lines_full, primary_line_times_full = concat_timed_line_chunks(
        [frame.segments for frame in primary_lines_by_frame if len(frame.segments) > 0],
        [frame.colors for frame in primary_lines_by_frame if len(frame.colors) > 0],
        [times for times in primary_line_times_by_frame if len(times) > 0],
        [frame.seed_border_dist_px for frame in primary_lines_by_frame if len(frame.seed_border_dist_px) > 0],
        [frame.in_bounds_ratio for frame in primary_lines_by_frame if len(frame.in_bounds_ratio) > 0],
        [frame.max_uv_step_px for frame in primary_lines_by_frame if len(frame.max_uv_step_px) > 0],
        max_segments=max_full_segments,
    )
    secondary_lines_full, secondary_line_times_full = concat_timed_line_chunks(
        [frame.segments for frame in secondary_lines_by_frame if len(frame.segments) > 0],
        [frame.colors for frame in secondary_lines_by_frame if len(frame.colors) > 0],
        [times for times in secondary_line_times_by_frame if len(times) > 0],
        [frame.seed_border_dist_px for frame in secondary_lines_by_frame if len(frame.seed_border_dist_px) > 0],
        [frame.in_bounds_ratio for frame in secondary_lines_by_frame if len(frame.in_bounds_ratio) > 0],
        [frame.max_uv_step_px for frame in secondary_lines_by_frame if len(frame.max_uv_step_px) > 0],
        max_segments=max_full_segments,
    )

    total_primary_points = int(sum(len(frame.points) for frame in primary_by_frame))
    total_secondary_points = int(sum(len(frame.points) for frame in secondary_by_frame))
    total_primary_segments = int(sum(len(frame.segments) for frame in primary_lines_by_frame))
    total_secondary_segments = int(sum(len(frame.segments) for frame in secondary_lines_by_frame))
    return AggregatedTracks(
        primary_by_frame=primary_by_frame,
        secondary_by_frame=secondary_by_frame,
        primary_lines_by_frame=primary_lines_by_frame,
        secondary_lines_by_frame=secondary_lines_by_frame,
        primary_lines_full=primary_lines_full,
        secondary_lines_full=secondary_lines_full,
        primary_line_times_full=primary_line_times_full,
        secondary_line_times_full=secondary_line_times_full,
        query_frames=selected_query_frames,
        sample_count=len(selected_query_frames),
        total_tracks_after_stride=total_tracks_after_stride,
        total_primary_points=total_primary_points,
        total_secondary_points=total_secondary_points,
        total_primary_segments=total_primary_segments,
        total_secondary_segments=total_secondary_segments,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive 4D reconstruction viewer")
    parser.add_argument(
        "--episode_dir",
        type=str,
        required=True,
        help="Camera directory that contains scene artifacts and sample NPZs.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=RENDER_MODE_FINITE,
        choices=RENDER_MODES,
        help="Track render mode for aggregated samples.",
    )
    parser.add_argument(
        "--query_stride",
        type=int,
        default=1,
        help="Use every N-th query sample when aggregating tracked points.",
    )
    parser.add_argument(
        "--track_stride",
        type=int,
        default=1,
        help="Use every N-th track inside each sample NPZ.",
    )
    parser.add_argument(
        "--max_points_per_frame",
        type=int,
        default=50000,
        help="Cap the displayed tracked points per frame after aggregation.",
    )
    parser.add_argument(
        "--max_segments_per_frame",
        type=int,
        default=25000,
        help="Cap the displayed trajectory segments contributed by each global frame.",
    )
    parser.add_argument(
        "--max_full_segments",
        type=int,
        default=200000,
        help="Cap the displayed full-trajectory segments when using complete trajectory mode.",
    )
    parser.add_argument(
        "--dense_downsample",
        type=int,
        default=4,
        help="Dense point cloud downsample factor.",
    )
    parser.add_argument(
        "--max_dense_points_per_frame",
        type=int,
        default=DEFAULT_MAX_DENSE_POINTS_PER_FRAME,
        help="Cap the displayed dense point cloud points per frame after reconstruction. <=0 disables the cap.",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        default=COLOR_MODE_TURBO,
        choices=COLOR_MODES,
        help="Tracked point coloring mode.",
    )
    parser.add_argument(
        "--buffer_mode",
        type=str,
        default=BUFFER_MODE_AUTO,
        choices=BUFFER_MODES,
        help="Playback mode: auto buffers small episodes and streams larger ones.",
    )
    parser.add_argument(
        "--preload_dense",
        action="store_true",
        help="Precompute dense point clouds for all frames before opening the viewer.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser port.",
    )
    return parser


def resolve_episode_camera_dir(raw_episode_dir: Path) -> Path:
    raw_episode_dir = raw_episode_dir.resolve()
    if not raw_episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {raw_episode_dir}")

    if (raw_episode_dir / "samples").is_dir():
        return raw_episode_dir

    candidates: list[Path] = []
    for child in sorted(raw_episode_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "samples").is_dir():
            candidates.append(child)

    if len(candidates) == 1:
        logger.info("Resolved camera directory: {} -> {}", raw_episode_dir, candidates[0])
        return candidates[0]

    if len(candidates) > 1:
        candidate_str = ", ".join(str(path.name) for path in candidates)
        raise ValueError(
            f"{raw_episode_dir} looks like a trajectory root with multiple camera subdirectories: {candidate_str}. "
            "Please pass a specific camera directory."
        )

    raise FileNotFoundError(
        f"No sample NPZ files found under {raw_episode_dir}. "
        "Please pass a camera directory like .../trajectory_xxx/<camera_name>."
    )


def main() -> None:
    args = build_parser().parse_args()
    episode_dir = resolve_episode_camera_dir(Path(args.episode_dir))

    with SceneReader(episode_dir) as scene_reader:
        intrinsics, extrinsics = scene_reader.get_camera_arrays()
        total_frames = int(len(intrinsics))
        if total_frames <= 0:
            raise ValueError(f"No frames found for {episode_dir}")

        aggregated = aggregate_tracked_points(
            episode_dir=episode_dir,
            scene_reader=scene_reader,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            render_mode=str(args.render_mode),
            query_stride=int(args.query_stride),
            track_stride=int(args.track_stride),
            max_points_per_frame=int(args.max_points_per_frame),
            max_segments_per_frame=int(args.max_segments_per_frame),
            max_full_segments=int(args.max_full_segments),
            color_mode=str(args.color_mode),
        )

        rgb_cache: dict[int, np.ndarray] = {}
        dense_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def get_rgb(frame_idx: int) -> np.ndarray:
            frame_idx = int(frame_idx)
            if frame_idx not in rgb_cache:
                rgb_cache[frame_idx] = scene_reader.get_rgb_frame(frame_idx)
            return rgb_cache[frame_idx]

        playback_mode = resolve_buffer_mode(str(args.buffer_mode), total_frames)
        logger.info(
            "4D viewer playback mode: {} (frames={}, max_dense_points_per_frame={})",
            playback_mode,
            total_frames,
            int(args.max_dense_points_per_frame),
        )

        def get_dense(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
            frame_idx = int(frame_idx)
            if frame_idx not in dense_cache:
                dense_cache[frame_idx] = build_dense_pointcloud_for_frame(
                    scene_reader=scene_reader,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    frame_idx=frame_idx,
                    downsample=int(args.dense_downsample),
                    max_points=int(args.max_dense_points_per_frame),
                )
            return dense_cache[frame_idx]

        if args.preload_dense or playback_mode == BUFFER_MODE_BUFFERED:
            logger.info("Preloading dense point clouds for {} frames...", total_frames)
            for frame_idx in range(total_frames):
                _ = get_dense(frame_idx)

        server = viser.ViserServer(port=int(args.port))
        server.scene.set_up_direction("-y")

        initial_frame_idx = 1 if total_frames > 1 else 0
        initial_rgb = get_rgb(initial_frame_idx)
        initial_display_settings = DisplaySettings(
            track_downsample=DEFAULT_DISPLAY_DOWNSAMPLE,
            dense_downsample=DEFAULT_DISPLAY_DOWNSAMPLE,
            trail_downsample=DEFAULT_DISPLAY_DOWNSAMPLE,
            trail_mode=TRAIL_MODE_PROGRESSIVE,
            fade_trails=True,
            fade_frames=min(max(total_frames, 1), DEFAULT_TRAIL_FADE_FRAMES),
            trail_window=min(total_frames, 12),
            seed_border_margin_px=DEFAULT_TRACK_SEED_BORDER_MARGIN_PX,
            min_in_bounds_ratio=DEFAULT_TRACK_MIN_IN_BOUNDS_RATIO,
            max_uv_step_px=DEFAULT_TRACK_MAX_UV_STEP_PX,
        )
        initial_display_data = build_display_frame_data(
            frame_idx=initial_frame_idx,
            aggregated=aggregated,
            get_dense=get_dense,
            settings=initial_display_settings,
        )

        camera_frustums: list[viser.CameraFrustumHandle] = []
        h, w = int(initial_rgb.shape[0]), int(initial_rgb.shape[1])
        for frame_idx in range(total_frames):
            rgb = get_rgb(frame_idx)
            c2w = np.linalg.inv(extrinsics[frame_idx]).astype(np.float32)
            fov = 2.0 * math.atan2(float(h) * 0.5, float(intrinsics[frame_idx][0, 0]))
            aspect = float(w) / max(float(h), 1.0)
            frustum = server.scene.add_camera_frustum(
                name=f"/frustums/{frame_idx:04d}",
                fov=fov,
                aspect=aspect,
                scale=0.12,
                image=rgb[::4, ::4],
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )
            camera_frustums.append(frustum)

        with server.gui.add_folder("4D Reconstruction"):
            gui_time = server.gui.add_slider(
                "Frame",
                min=0,
                max=max(total_frames - 1, 0),
                step=1,
                initial_value=initial_frame_idx,
            )
            gui_play = server.gui.add_checkbox("Play", playback_mode == BUFFER_MODE_BUFFERED)
            gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=DEFAULT_PLAYBACK_FPS)
            gui_show_track_points = server.gui.add_checkbox("Show Track Points", True)
            gui_show_dense = server.gui.add_checkbox("Show Dense Point Cloud", True)
            gui_show_trails = server.gui.add_checkbox("Show Trajectory Lines", True)
            gui_show_frustums = server.gui.add_checkbox("Show Frustums", False)
            gui_track_size = server.gui.add_slider(
                "Track Point Size",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=DEFAULT_TRACK_POINT_SIZE,
            )
            gui_dense_size = server.gui.add_slider(
                "Dense Point Size",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=DEFAULT_DENSE_POINT_SIZE,
            )
            gui_trail_width = server.gui.add_slider("Trajectory Line Width", min=0.5, max=8.0, step=0.5, initial_value=2.0)
            gui_frame_track_count = server.gui.add_number(
                "Frame Track Points",
                initial_value=int(len(initial_display_data.primary.points) + len(initial_display_data.secondary.points)),
                disabled=True,
            )

        with server.gui.add_folder("Current Frame"):
            gui_rgb = server.gui.add_image(
                prepare_rgb_preview(initial_rgb, DEFAULT_RGB_PREVIEW_DOWNSAMPLE),
                label="RGB",
            )

        tracked_primary_handle = None
        tracked_secondary_handle = None
        dense_handle = None
        primary_trail_handle = None
        secondary_trail_handle = None
        buffered_handles: list[BufferedFrameHandles] = []
        buffered_display_cache: list[DisplayFrameData] = []
        buffered_settings: DisplaySettings | None = None

        if playback_mode == BUFFER_MODE_STREAM:
            tracked_primary_handle = server.scene.add_point_cloud(
                name="tracks_primary",
                points=initial_display_data.primary.points,
                colors=initial_display_data.primary.colors,
                point_size=DEFAULT_TRACK_POINT_SIZE,
                point_shape="rounded",
                precision="float32",
            )
            tracked_secondary_handle = server.scene.add_point_cloud(
                name="tracks_secondary",
                points=initial_display_data.secondary.points,
                colors=initial_display_data.secondary.colors,
                point_size=DEFAULT_TRACK_POINT_SIZE * DEFAULT_SECONDARY_TRACK_SIZE_RATIO,
                point_shape="rounded",
                precision="float32",
            )
            dense_handle = server.scene.add_point_cloud(
                name="dense_pointcloud",
                points=initial_display_data.dense.points,
                colors=initial_display_data.dense.colors,
                point_size=DEFAULT_DENSE_POINT_SIZE,
                point_shape="rounded",
                precision="float32",
            )
            primary_trail_handle = server.scene.add_line_segments(
                name="tracks_primary_lines",
                points=initial_display_data.primary_lines.segments,
                colors=initial_display_data.primary_lines.colors,
                line_width=2.0,
            )
            secondary_trail_handle = server.scene.add_line_segments(
                name="tracks_secondary_lines",
                points=initial_display_data.secondary_lines.segments,
                colors=initial_display_data.secondary_lines.colors,
                line_width=1.5,
            )

        def current_display_settings() -> DisplaySettings:
            return DisplaySettings(
                track_downsample=initial_display_settings.track_downsample,
                dense_downsample=initial_display_settings.dense_downsample,
                trail_downsample=initial_display_settings.trail_downsample,
                trail_mode=initial_display_settings.trail_mode,
                fade_trails=initial_display_settings.fade_trails,
                fade_frames=initial_display_settings.fade_frames,
                trail_window=initial_display_settings.trail_window,
                seed_border_margin_px=initial_display_settings.seed_border_margin_px,
                min_in_bounds_ratio=initial_display_settings.min_in_bounds_ratio,
                max_uv_step_px=initial_display_settings.max_uv_step_px,
            )

        def update_frustums(frame_idx: int) -> None:
            for idx, frustum in enumerate(camera_frustums):
                frustum.scale = 0.12
                frustum.visible = gui_show_frustums.value and idx == frame_idx

        def update_rgb_preview(frame_idx: int, *, force: bool = False) -> None:
            gui_rgb.image = prepare_rgb_preview(
                get_rgb(int(frame_idx)),
                DEFAULT_RGB_PREVIEW_DOWNSAMPLE,
            )

        def apply_stream_scene(frame_idx: int, display_data: DisplayFrameData) -> None:
            assert tracked_primary_handle is not None
            assert tracked_secondary_handle is not None
            assert dense_handle is not None
            assert primary_trail_handle is not None
            assert secondary_trail_handle is not None

            tracked_primary_handle.points = display_data.primary.points
            tracked_primary_handle.colors = display_data.primary.colors
            tracked_primary_handle.point_size = gui_track_size.value

            tracked_secondary_handle.points = display_data.secondary.points
            tracked_secondary_handle.colors = display_data.secondary.colors
            tracked_secondary_handle.point_size = max(
                0.001,
                gui_track_size.value * DEFAULT_SECONDARY_TRACK_SIZE_RATIO,
            )

            dense_handle.points = display_data.dense.points
            dense_handle.colors = display_data.dense.colors
            dense_handle.point_size = gui_dense_size.value

            primary_trail_handle.points = display_data.primary_lines.segments
            primary_trail_handle.colors = display_data.primary_lines.colors
            primary_trail_handle.line_width = gui_trail_width.value

            secondary_trail_handle.points = display_data.secondary_lines.segments
            secondary_trail_handle.colors = display_data.secondary_lines.colors
            secondary_trail_handle.line_width = max(0.5, gui_trail_width.value * 0.75)

            show_track_points = bool(gui_show_track_points.value)
            show_dense = bool(gui_show_dense.value)
            tracked_primary_handle.visible = show_track_points
            tracked_secondary_handle.visible = (
                show_track_points
                and str(args.render_mode) == RENDER_MODE_HYBRID
                and len(display_data.secondary.points) > 0
            )
            dense_handle.visible = show_dense
            primary_trail_handle.visible = (
                gui_show_trails.value
                and len(display_data.primary_lines.segments) > 0
            )
            secondary_trail_handle.visible = (
                gui_show_trails.value
                and str(args.render_mode) == RENDER_MODE_HYBRID
                and len(display_data.secondary_lines.segments) > 0
            )

            gui_frame_track_count.value = int(len(display_data.primary.points) + len(display_data.secondary.points))

        def apply_buffered_visibility(frame_idx: int) -> None:
            if not buffered_handles:
                return
            current_idx = int(frame_idx)
            show_track_points = bool(gui_show_track_points.value)
            show_dense = bool(gui_show_dense.value)
            for idx, handles in enumerate(buffered_handles):
                display_data = buffered_display_cache[idx]
                is_current = idx == current_idx
                handles.primary.point_size = gui_track_size.value
                handles.secondary.point_size = max(
                    0.001,
                    gui_track_size.value * DEFAULT_SECONDARY_TRACK_SIZE_RATIO,
                )
                handles.dense.point_size = gui_dense_size.value
                handles.primary_lines.line_width = gui_trail_width.value
                handles.secondary_lines.line_width = max(0.5, gui_trail_width.value * 0.75)
                handles.primary.visible = is_current and show_track_points
                handles.secondary.visible = (
                    is_current
                    and show_track_points
                    and str(args.render_mode) == RENDER_MODE_HYBRID
                    and len(display_data.secondary.points) > 0
                )
                handles.dense.visible = is_current and show_dense
                handles.primary_lines.visible = (
                    is_current
                    and gui_show_trails.value
                    and len(display_data.primary_lines.segments) > 0
                )
                handles.secondary_lines.visible = (
                    is_current
                    and gui_show_trails.value
                    and str(args.render_mode) == RENDER_MODE_HYBRID
                    and len(display_data.secondary_lines.segments) > 0
                )
            current_display = buffered_display_cache[current_idx]
            gui_frame_track_count.value = int(len(current_display.primary.points) + len(current_display.secondary.points))

        def rebuild_buffered_scene(*, force: bool = False) -> None:
            nonlocal buffered_settings, buffered_display_cache
            settings = current_display_settings()
            if not force and buffered_settings == settings and buffered_display_cache:
                return

            buffered_display_cache = []
            for frame_idx in range(total_frames):
                display_data = build_display_frame_data(
                    frame_idx=frame_idx,
                    aggregated=aggregated,
                    get_dense=get_dense,
                    settings=settings,
                )
                buffered_display_cache.append(display_data)
                if frame_idx >= len(buffered_handles):
                    handles = BufferedFrameHandles(
                        primary=server.scene.add_point_cloud(
                            name=f"/buffered/{frame_idx:04d}/tracks_primary",
                            points=display_data.primary.points,
                            colors=display_data.primary.colors,
                            point_size=DEFAULT_TRACK_POINT_SIZE,
                            point_shape="rounded",
                            precision="float32",
                        ),
                        secondary=server.scene.add_point_cloud(
                            name=f"/buffered/{frame_idx:04d}/tracks_secondary",
                            points=display_data.secondary.points,
                            colors=display_data.secondary.colors,
                            point_size=DEFAULT_TRACK_POINT_SIZE * DEFAULT_SECONDARY_TRACK_SIZE_RATIO,
                            point_shape="rounded",
                            precision="float32",
                        ),
                        dense=server.scene.add_point_cloud(
                            name=f"/buffered/{frame_idx:04d}/dense_pointcloud",
                            points=display_data.dense.points,
                            colors=display_data.dense.colors,
                            point_size=DEFAULT_DENSE_POINT_SIZE,
                            point_shape="rounded",
                            precision="float32",
                        ),
                        primary_lines=server.scene.add_line_segments(
                            name=f"/buffered/{frame_idx:04d}/tracks_primary_lines",
                            points=display_data.primary_lines.segments,
                            colors=display_data.primary_lines.colors,
                            line_width=2.0,
                        ),
                        secondary_lines=server.scene.add_line_segments(
                            name=f"/buffered/{frame_idx:04d}/tracks_secondary_lines",
                            points=display_data.secondary_lines.segments,
                            colors=display_data.secondary_lines.colors,
                            line_width=1.5,
                        ),
                    )
                    buffered_handles.append(handles)
                else:
                    handles = buffered_handles[frame_idx]
                    handles.primary.points = display_data.primary.points
                    handles.primary.colors = display_data.primary.colors
                    handles.secondary.points = display_data.secondary.points
                    handles.secondary.colors = display_data.secondary.colors
                    handles.dense.points = display_data.dense.points
                    handles.dense.colors = display_data.dense.colors
                    handles.primary_lines.points = display_data.primary_lines.segments
                    handles.primary_lines.colors = display_data.primary_lines.colors
                    handles.secondary_lines.points = display_data.secondary_lines.segments
                    handles.secondary_lines.colors = display_data.secondary_lines.colors
            buffered_settings = settings
            apply_buffered_visibility(int(gui_time.value))

        def update_display(*, force_rgb: bool = False) -> None:
            frame_idx = int(gui_time.value)
            if playback_mode == BUFFER_MODE_BUFFERED:
                rebuild_buffered_scene()
                apply_buffered_visibility(frame_idx)
            else:
                display_data = build_display_frame_data(
                    frame_idx=frame_idx,
                    aggregated=aggregated,
                    get_dense=get_dense,
                    settings=current_display_settings(),
                )
                apply_stream_scene(frame_idx, display_data)
            update_rgb_preview(frame_idx, force=force_rgb)
            update_frustums(frame_idx)

        @gui_time.on_update
        def _(_) -> None:
            update_display()

        @gui_play.on_update
        def _(_) -> None:
            update_display(force_rgb=True)

        @gui_show_track_points.on_update
        def _(_) -> None:
            update_display()

        @gui_show_dense.on_update
        def _(_) -> None:
            update_display()

        @gui_track_size.on_update
        def _(_) -> None:
            update_display()

        @gui_dense_size.on_update
        def _(_) -> None:
            update_display()

        @gui_show_trails.on_update
        def _(_) -> None:
            update_display()

        @gui_trail_width.on_update
        def _(_) -> None:
            update_display()

        @gui_show_frustums.on_update
        def _(_) -> None:
            update_frustums(int(gui_time.value))

        update_display(force_rgb=True)

        logger.info(
            "Loaded 4D viewer: frames={}, query_samples={}, query_frames={}, tracks_after_stride={}, "
            "primary_points={}, secondary_points={}, primary_segments={}, secondary_segments={}",
            total_frames,
            aggregated.sample_count,
            aggregated.query_frames,
            aggregated.total_tracks_after_stride,
            aggregated.total_primary_points,
            aggregated.total_secondary_points,
            aggregated.total_primary_segments,
            aggregated.total_secondary_segments,
        )
        logger.info("Viser server: http://localhost:{}", args.port)

        try:
            while True:
                if gui_play.value and total_frames > 1:
                    gui_time.value = (int(gui_time.value) + 1) % total_frames
                time.sleep(1.0 / max(int(gui_fps.value), 1))
        except KeyboardInterrupt:
            logger.info("Exit")


if __name__ == "__main__":
    main()
