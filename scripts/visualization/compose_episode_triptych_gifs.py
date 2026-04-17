#!/usr/bin/env python3
"""
Export triptych GIFs with dynamic RGB, configurable 2D backgrounds, and 3D trajectories.

This script is visualization-only. It does not change saved trajectory samples.
It supports a lightweight outlier cull so a few runaway tracks do not dominate
the rendered GIFs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import verify_episode_trajectory_outputs as verify
from utils.traceforge_artifact_utils import (
    SceneReader,
    build_sample_visualization_view,
    normalize_sample_data,
    traj_uvz_to_world,
)


RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
DEFAULT_PANEL_HEIGHT = 360
TRACKS_2D_BACKGROUND_DYNAMIC = "dynamic"
TRACKS_2D_BACKGROUND_STATIC = "static"
TRACKS_2D_BACKGROUND_QUERY_STATIC = "query_static"
TRACKS_2D_BACKGROUND_MODES = (
    TRACKS_2D_BACKGROUND_DYNAMIC,
    TRACKS_2D_BACKGROUND_STATIC,
    TRACKS_2D_BACKGROUND_QUERY_STATIC,
)


@dataclass(frozen=True)
class QueryTriptychArtifact:
    query_frame: int
    segment_frame_indices: list[int]
    visualized_track_count: int
    rgb_gif_path: str
    tracks_2d_gif_path: str
    tracks_3d_gif_path: str
    composite_gif_path: str
    frame_count: int


def parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return [int(item) for item in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render RGB + 2D + 3D triptych GIFs for episode trajectory outputs."
    )
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--trajectory_dirname", type=str, default="trajectory")
    parser.add_argument("--camera_name", type=str, required=True)
    parser.add_argument("--query_frames", type=str, required=True, help="Comma-separated query frame indices.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--render_mode",
        type=str,
        default=verify.RENDER_MODE_FINITE,
        choices=verify.RENDER_MODES,
        help="Primary render mode used before visualization-only outlier culling.",
    )
    parser.add_argument("--gif_fps", type=int, default=6)
    parser.add_argument("--gif_dpi", type=int, default=100)
    parser.add_argument("--panel_height", type=int, default=DEFAULT_PANEL_HEIGHT)
    parser.add_argument(
        "--2d_bg_mode",
        dest="tracks_2d_background",
        type=str,
        default=TRACKS_2D_BACKGROUND_STATIC,
        choices=(
            TRACKS_2D_BACKGROUND_DYNAMIC,
            TRACKS_2D_BACKGROUND_STATIC,
        ),
        help="Background mode for the middle 2D panel. Defaults to static.",
    )
    parser.add_argument(
        "--tracks_2d_background",
        type=str,
        default=TRACKS_2D_BACKGROUND_STATIC,
        choices=TRACKS_2D_BACKGROUND_MODES,
        help="Deprecated compatibility alias for --2d_bg_mode. Supports static, dynamic, and query_static.",
    )
    parser.add_argument("--max_tracks", type=int, default=400)
    parser.add_argument("--max_cloud_points", type=int, default=4000)
    parser.add_argument("--ply_downsample", type=int, default=4)
    parser.add_argument("--depth_min", type=float, default=0.01)
    parser.add_argument("--depth_max", type=float, default=10.0)
    parser.add_argument("--line_alpha", type=float, default=0.9)
    parser.add_argument("--line_width", type=float, default=1.2)
    parser.add_argument(
        "--min_in_bounds_frames",
        type=int,
        default=8,
        help="Visualization-only keep rule: minimum number of in-bounds frames.",
    )
    parser.add_argument(
        "--max_overflow_px",
        type=float,
        default=128.0,
        help="Visualization-only keep rule: maximum allowed out-of-frame excursion in pixels.",
    )
    parser.add_argument(
        "--allow_query_out_of_bounds",
        action="store_true",
        help="Do not require the query-frame point to start inside the image.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="summary.json",
        help="Summary JSON filename written under output_dir.",
    )
    return parser


def add_label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    box_height = 24
    box_width = min(labeled.width, max(320, 7 * len(text)))
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    return labeled


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    target_height = max(int(target_height), 1)
    if image.height == target_height:
        return image.copy()
    target_width = max(int(round(image.width * target_height / image.height)), 1)
    return image.resize((target_width, target_height), RESAMPLE_LANCZOS)


def save_gif(output_path: Path, frames: list[Image.Image], fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {output_path}")
    verify.save_gif(output_path, frames, fps)


def normalize_tracks_2d_background_mode(background_mode: str) -> str:
    if background_mode == TRACKS_2D_BACKGROUND_QUERY_STATIC:
        return TRACKS_2D_BACKGROUND_STATIC
    if background_mode not in {
        TRACKS_2D_BACKGROUND_DYNAMIC,
        TRACKS_2D_BACKGROUND_STATIC,
    }:
        raise ValueError(f"Unsupported tracks_2d_background mode: {background_mode}")
    return background_mode


def tracks_2d_dirname(background_mode: str) -> str:
    if background_mode == TRACKS_2D_BACKGROUND_STATIC:
        return "tracks_2d_query_static_gifs"
    return "tracks_2d_dynamic_gifs"


def tracks_2d_filename_suffix(background_mode: str) -> str:
    if background_mode == TRACKS_2D_BACKGROUND_STATIC:
        return "2d_query_static"
    return "2d_dynamic"


def composite_filename_suffix(background_mode: str) -> str:
    if background_mode == TRACKS_2D_BACKGROUND_STATIC:
        return "rgb_2d_query_static_3d"
    return "rgb_2d_3d"


def compute_visualization_keep_mask(
    *,
    traj_uvz_finite: np.ndarray,
    image_height: int,
    image_width: int,
    min_in_bounds_frames: int,
    max_overflow_px: float,
    require_query_in_bounds: bool,
) -> tuple[np.ndarray, dict[str, int | float]]:
    traj_uvz_finite = np.asarray(traj_uvz_finite, dtype=np.float32)
    finite = np.isfinite(traj_uvz_finite).all(axis=-1) & (traj_uvz_finite[..., 2] > 0)
    u_values = traj_uvz_finite[..., 0]
    v_values = traj_uvz_finite[..., 1]
    in_bounds = (
        finite
        & (u_values >= 0.0)
        & (u_values < float(image_width))
        & (v_values >= 0.0)
        & (v_values < float(image_height))
    )
    overflow = np.maximum.reduce(
        [
            np.where(finite, np.maximum(0.0, -u_values), 0.0),
            np.where(finite, np.maximum(0.0, u_values - float(image_width - 1)), 0.0),
            np.where(finite, np.maximum(0.0, -v_values), 0.0),
            np.where(finite, np.maximum(0.0, v_values - float(image_height - 1)), 0.0),
        ]
    )
    track_max_overflow = overflow.max(axis=1)
    in_bounds_counts = in_bounds.sum(axis=1).astype(np.int32)

    keep_mask = np.ones(traj_uvz_finite.shape[0], dtype=bool)
    if require_query_in_bounds:
        keep_mask &= in_bounds[:, 0]
    keep_mask &= in_bounds_counts >= int(min_in_bounds_frames)
    keep_mask &= track_max_overflow <= float(max_overflow_px)

    stats = {
        "raw_track_count": int(traj_uvz_finite.shape[0]),
        "kept_track_count": int(keep_mask.sum()),
        "require_query_in_bounds": int(require_query_in_bounds),
        "min_in_bounds_frames": int(min_in_bounds_frames),
        "max_overflow_px": float(max_overflow_px),
        "query_in_bounds_count": int(in_bounds[:, 0].sum()),
        "in_bounds_at_least_min_count": int((in_bounds_counts >= int(min_in_bounds_frames)).sum()),
        "overflow_within_limit_count": int((track_max_overflow <= float(max_overflow_px)).sum()),
    }
    return keep_mask, stats


def choose_filtered_track_indices(
    *,
    traj_world_selection: np.ndarray,
    keep_mask: np.ndarray,
    max_tracks: int,
) -> np.ndarray:
    keep_mask = np.asarray(keep_mask, dtype=bool)
    candidate_indices = np.flatnonzero(keep_mask)
    if candidate_indices.size == 0:
        return candidate_indices.astype(np.int32)
    ranked_local = verify.choose_track_indices(traj_world_selection[candidate_indices], max_tracks)
    return candidate_indices[ranked_local].astype(np.int32, copy=False)


def render_rgb_frames(
    *,
    aligned_rgb_frames: list[np.ndarray],
    segment_frame_indices: np.ndarray,
    query_frame: int,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for source_frame_idx, rgb in zip(segment_frame_indices.tolist(), aligned_rgb_frames):
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        frames.append(add_label(image, f"RGB frame={source_frame_idx:05d} query={query_frame:05d}"))
    return frames


def render_2d_frames(
    *,
    camera_name: str,
    query_frame: int,
    aligned_rgb_frames: list[np.ndarray],
    segment_frame_indices: np.ndarray,
    traj_2d_primary: np.ndarray,
    traj_2d_secondary: np.ndarray,
    track_indices: np.ndarray,
    line_alpha: float,
    line_width: float,
    gif_dpi: int,
    background_mode: str,
) -> list[Image.Image]:
    selected_traj_2d = traj_2d_primary[track_indices]
    selected_traj_2d_secondary = traj_2d_secondary[track_indices]
    track_colors = verify.make_track_colors(len(track_indices))
    frames: list[Image.Image] = []

    for t, (source_frame_idx, rgb) in enumerate(zip(segment_frame_indices.tolist(), aligned_rgb_frames)):
        background_rgb = aligned_rgb_frames[0] if background_mode == TRACKS_2D_BACKGROUND_STATIC else rgb
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ax.imshow(np.asarray(background_rgb, dtype=np.uint8))
        verify._plot_2d_trajectories(
            ax=ax,
            traj_primary=selected_traj_2d,
            traj_secondary=selected_traj_2d_secondary,
            track_colors=track_colors,
            line_alpha=line_alpha,
            line_width=line_width,
            time_limit=t,
        )
        ax.set_title(
            f"{camera_name} | query frame {query_frame} | frame {source_frame_idx} | "
            f"t={t} | 2D trajectories | bg={background_mode}"
        )
        ax.set_axis_off()
        frames.append(verify.figure_to_image(fig, dpi=gif_dpi))
    return frames


def render_3d_frames(
    *,
    camera_name: str,
    query_frame: int,
    segment_frame_indices: np.ndarray,
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    traj_world_primary: np.ndarray,
    traj_world_secondary: np.ndarray,
    track_indices: np.ndarray,
    max_cloud_points: int,
    line_alpha: float,
    line_width: float,
    gif_dpi: int,
) -> list[Image.Image]:
    selected_traj_world = traj_world_primary[track_indices]
    selected_traj_world_secondary = traj_world_secondary[track_indices]
    track_colors = verify.make_track_colors(len(track_indices))
    cloud_points_plot, cloud_colors_plot = verify.sample_cloud_points(
        cloud_points,
        cloud_colors,
        max_cloud_points,
    )
    traj_points = np.concatenate(
        [selected_traj_world.reshape(-1, 3), selected_traj_world_secondary.reshape(-1, 3)],
        axis=0,
    )
    center, radius = verify.build_axis_limits(np.concatenate([cloud_points_plot, traj_points], axis=0))

    frames: list[Image.Image] = []
    num_frames = selected_traj_world.shape[1]
    for t in range(num_frames):
        source_frame_idx = int(segment_frame_indices[t])
        fig = plt.figure(figsize=(6, 5), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(
            cloud_points_plot[:, 0],
            cloud_points_plot[:, 1],
            cloud_points_plot[:, 2],
            c=cloud_colors_plot,
            s=0.4,
            alpha=0.7,
            linewidths=0.0,
        )
        verify._plot_3d_trajectories(
            ax=ax,
            traj_primary=selected_traj_world,
            traj_secondary=selected_traj_world_secondary,
            track_colors=track_colors,
            line_alpha=line_alpha,
            line_width=line_width,
            time_limit=t,
        )
        verify.apply_axis_limits(ax, center, radius)
        ax.set_title(
            f"{camera_name} | query frame {query_frame} | frame {source_frame_idx} | t={t} | world trajectories"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-58)
        frames.append(verify.figure_to_image(fig, dpi=gif_dpi))
    return frames


def compose_triptych_frames(
    *,
    rgb_frames: list[Image.Image],
    frames_2d: list[Image.Image],
    frames_3d: list[Image.Image],
    panel_height: int,
    query_frame: int,
) -> list[Image.Image]:
    frame_count = min(len(rgb_frames), len(frames_2d), len(frames_3d))
    composite_frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        panel_rgb = resize_to_height(rgb_frames[frame_idx], panel_height)
        panel_2d = resize_to_height(frames_2d[frame_idx], panel_height)
        panel_3d = resize_to_height(frames_3d[frame_idx], panel_height)
        total_width = panel_rgb.width + panel_2d.width + panel_3d.width
        canvas = Image.new("RGB", (total_width, panel_height), color=(255, 255, 255))
        x = 0
        for panel in (panel_rgb, panel_2d, panel_3d):
            canvas.paste(panel, (x, 0))
            x += panel.width
        composite_frames.append(add_label(canvas, f"query={query_frame:05d} t={frame_idx:02d}"))
    return composite_frames


def build_query_bundle(
    *,
    episode_dir: Path,
    trajectory_dirname: str,
    camera_name: str,
    query_frame: int,
    render_mode: str,
    ply_downsample: int,
    depth_min: float,
    depth_max: float,
    min_in_bounds_frames: int,
    max_overflow_px: float,
    require_query_in_bounds: bool,
    max_tracks: int,
) -> dict:
    camera_dir = episode_dir / trajectory_dirname / camera_name
    sample_path = camera_dir / "samples" / f"{camera_name}_{query_frame}.npz"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing sample NPZ: {sample_path}")

    sample = normalize_sample_data(sample_path)
    render_view = build_sample_visualization_view(sample, render_mode=render_mode)
    segment_frame_indices = np.asarray(render_view["segment_frame_indices"], dtype=np.int32)

    with SceneReader(camera_dir) as scene_reader:
        intrinsics, extrinsics = scene_reader.get_camera_arrays()
        aligned_rgb_frames = [scene_reader.get_rgb_frame(int(frame_idx)) for frame_idx in segment_frame_indices.tolist()]

    query_rgb = np.asarray(aligned_rgb_frames[0], dtype=np.uint8)
    image_height, image_width = int(query_rgb.shape[0]), int(query_rgb.shape[1])
    traj_uvz_selection = np.asarray(render_view["traj_uvz_finite"], dtype=np.float32)
    keep_mask, keep_stats = compute_visualization_keep_mask(
        traj_uvz_finite=traj_uvz_selection,
        image_height=image_height,
        image_width=image_width,
        min_in_bounds_frames=min_in_bounds_frames,
        max_overflow_px=max_overflow_px,
        require_query_in_bounds=require_query_in_bounds,
    )
    traj_world_selection = traj_uvz_to_world(
        traj_uvz_selection,
        intrinsics[query_frame].astype(np.float32),
        extrinsics[query_frame].astype(np.float32),
    )
    track_indices = choose_filtered_track_indices(
        traj_world_selection=traj_world_selection,
        keep_mask=keep_mask,
        max_tracks=max_tracks,
    )

    cloud_points, cloud_colors, _ = verify.create_pointcloud(
        camera_dir=camera_dir,
        camera_name=camera_name,
        frame_idx=query_frame,
        downsample=max(1, ply_downsample),
        depth_min=depth_min,
        depth_max=depth_max,
    )
    traj_world = traj_uvz_to_world(
        np.asarray(render_view["traj_uvz"], dtype=np.float32),
        intrinsics[query_frame].astype(np.float32),
        extrinsics[query_frame].astype(np.float32),
    )
    traj_world_secondary = traj_uvz_to_world(
        np.asarray(render_view["traj_uvz_secondary"], dtype=np.float32),
        intrinsics[query_frame].astype(np.float32),
        extrinsics[query_frame].astype(np.float32),
    )

    return {
        "camera_dir": camera_dir,
        "camera_name": camera_name,
        "query_frame": int(query_frame),
        "segment_frame_indices": segment_frame_indices,
        "aligned_rgb_frames": aligned_rgb_frames,
        "traj_2d": np.asarray(render_view["traj_2d"], dtype=np.float32),
        "traj_2d_secondary": np.asarray(render_view["traj_2d_secondary"], dtype=np.float32),
        "traj_world": traj_world,
        "traj_world_secondary": traj_world_secondary,
        "cloud_points": cloud_points,
        "cloud_colors": cloud_colors,
        "track_indices": track_indices,
        "keep_stats": keep_stats,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.tracks_2d_background = normalize_tracks_2d_background_mode(str(args.tracks_2d_background))

    episode_dir = args.episode_dir.resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb_gifs"
    tracks_2d_dir = output_dir / tracks_2d_dirname(str(args.tracks_2d_background))
    tracks_3d_dir = output_dir / "tracks_3d_gifs"
    composite_dir = output_dir / "composite_gifs"
    for path in (rgb_dir, tracks_2d_dir, tracks_3d_dir, composite_dir):
        path.mkdir(parents=True, exist_ok=True)

    query_frames = parse_csv_ints(args.query_frames)
    artifacts: list[QueryTriptychArtifact] = []
    keep_stats_by_query: dict[str, dict[str, int | float]] = {}

    for query_frame in query_frames:
        bundle = build_query_bundle(
            episode_dir=episode_dir,
            trajectory_dirname=str(args.trajectory_dirname),
            camera_name=str(args.camera_name),
            query_frame=int(query_frame),
            render_mode=str(args.render_mode),
            ply_downsample=max(1, int(args.ply_downsample)),
            depth_min=float(args.depth_min),
            depth_max=float(args.depth_max),
            min_in_bounds_frames=max(0, int(args.min_in_bounds_frames)),
            max_overflow_px=float(args.max_overflow_px),
            require_query_in_bounds=not bool(args.allow_query_out_of_bounds),
            max_tracks=max(1, int(args.max_tracks)),
        )
        if len(bundle["track_indices"]) == 0:
            raise ValueError(f"Visualization keep mask rejected all tracks for query_frame={query_frame}")

        rgb_frames = render_rgb_frames(
            aligned_rgb_frames=bundle["aligned_rgb_frames"],
            segment_frame_indices=bundle["segment_frame_indices"],
            query_frame=int(query_frame),
        )
        frames_2d = render_2d_frames(
            camera_name=str(args.camera_name),
            query_frame=int(query_frame),
            aligned_rgb_frames=bundle["aligned_rgb_frames"],
            segment_frame_indices=bundle["segment_frame_indices"],
            traj_2d_primary=bundle["traj_2d"],
            traj_2d_secondary=bundle["traj_2d_secondary"],
            track_indices=bundle["track_indices"],
            line_alpha=float(args.line_alpha),
            line_width=float(args.line_width),
            gif_dpi=max(60, int(args.gif_dpi)),
            background_mode=str(args.tracks_2d_background),
        )
        frames_3d = render_3d_frames(
            camera_name=str(args.camera_name),
            query_frame=int(query_frame),
            segment_frame_indices=bundle["segment_frame_indices"],
            cloud_points=bundle["cloud_points"],
            cloud_colors=bundle["cloud_colors"],
            traj_world_primary=bundle["traj_world"],
            traj_world_secondary=bundle["traj_world_secondary"],
            track_indices=bundle["track_indices"],
            max_cloud_points=max(1, int(args.max_cloud_points)),
            line_alpha=float(args.line_alpha),
            line_width=float(args.line_width),
            gif_dpi=max(60, int(args.gif_dpi)),
        )
        composite_frames = compose_triptych_frames(
            rgb_frames=rgb_frames,
            frames_2d=frames_2d,
            frames_3d=frames_3d,
            panel_height=max(120, int(args.panel_height)),
            query_frame=int(query_frame),
        )

        rgb_gif_path = rgb_dir / f"q{query_frame:05d}_rgb_segment.gif"
        tracks_2d_gif_path = (
            tracks_2d_dir / f"q{query_frame:05d}_{tracks_2d_filename_suffix(str(args.tracks_2d_background))}.gif"
        )
        tracks_3d_gif_path = tracks_3d_dir / f"q{query_frame:05d}_3d_tracks.gif"
        composite_gif_path = (
            composite_dir / f"q{query_frame:05d}_{composite_filename_suffix(str(args.tracks_2d_background))}.gif"
        )
        save_gif(rgb_gif_path, rgb_frames, max(1, int(args.gif_fps)))
        save_gif(tracks_2d_gif_path, frames_2d, max(1, int(args.gif_fps)))
        save_gif(tracks_3d_gif_path, frames_3d, max(1, int(args.gif_fps)))
        save_gif(composite_gif_path, composite_frames, max(1, int(args.gif_fps)))

        keep_stats_by_query[f"{int(query_frame):05d}"] = bundle["keep_stats"]
        artifact = QueryTriptychArtifact(
            query_frame=int(query_frame),
            segment_frame_indices=bundle["segment_frame_indices"].astype(int).tolist(),
            visualized_track_count=int(len(bundle["track_indices"])),
            rgb_gif_path=str(rgb_gif_path),
            tracks_2d_gif_path=str(tracks_2d_gif_path),
            tracks_3d_gif_path=str(tracks_3d_gif_path),
            composite_gif_path=str(composite_gif_path),
            frame_count=int(len(composite_frames)),
        )
        artifacts.append(artifact)
        print(
            f"[query={query_frame:05d}] tracks={artifact.visualized_track_count} "
            f"rgb={rgb_gif_path} gif2d={tracks_2d_gif_path} gif3d={tracks_3d_gif_path} composite={composite_gif_path}"
        )

    summary = {
        "episode_dir": str(episode_dir),
        "trajectory_dirname": str(args.trajectory_dirname),
        "camera_name": str(args.camera_name),
        "query_frames": [int(item) for item in query_frames],
        "render_mode": str(args.render_mode),
        "gif_fps": int(args.gif_fps),
        "gif_dpi": int(args.gif_dpi),
        "panel_height": int(args.panel_height),
        "tracks_2d_background": str(args.tracks_2d_background),
        "max_tracks": int(args.max_tracks),
        "max_cloud_points": int(args.max_cloud_points),
        "min_in_bounds_frames": int(args.min_in_bounds_frames),
        "max_overflow_px": float(args.max_overflow_px),
        "require_query_in_bounds": not bool(args.allow_query_out_of_bounds),
        "artifacts": [asdict(item) for item in artifacts],
        "keep_stats_by_query": keep_stats_by_query,
    }
    summary_path = output_dir / str(args.summary_name)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
