#!/usr/bin/env python3
"""
Batch-export side-by-side GIFs for sampled query frames.

For each query frame, this script renders:
1. An RGB GIF aligned to the sample segment_frame_indices.
2. A 2D trajectory overlay GIF in image space.
3. A 3D world-trajectory GIF with a point-cloud backdrop.

It then stitches the three panels horizontally into one composite GIF.

The existing query_rgb_scan/sampled_rgb_gifs directory is used to discover the
query-frame list, but the RGB panel is regenerated from segment_frame_indices so
that it stays time-aligned with the trajectory GIFs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

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


RGB_GIF_PATTERN = re.compile(r"^q(\d+)_rgb_window\.gif$")
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
DEFAULT_PANEL_HEIGHT = 360


@dataclass(frozen=True)
class QueryCompositeArtifact:
    query_frame: int
    segment_frame_indices: list[int]
    rgb_gif_path: str
    tracks_2d_gif_path: str
    tracks_3d_gif_path: str
    composite_gif_path: str
    frame_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render aligned RGB + 2D trajectory + 3D trajectory composite GIFs for sampled queries."
    )
    parser.add_argument(
        "--camera_dir",
        type=Path,
        required=True,
        help="Camera output root, e.g. /path/to/episode_00001/varied_camera_1",
    )
    parser.add_argument(
        "--sampled_rgb_gif_dir",
        type=Path,
        required=True,
        help="Directory containing qXXXXX_rgb_window.gif files. Used to discover query frames.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root for aligned RGB GIFs, trajectory GIFs, and composites.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=verify.RENDER_MODE_SUPERVISION,
        choices=verify.RENDER_MODES,
        help="Trajectory render mode for the 2D/3D GIFs.",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=10,
        help="Playback FPS for aligned RGB, trajectory, and composite GIFs.",
    )
    parser.add_argument(
        "--gif_dpi",
        type=int,
        default=90,
        help="Raster DPI for 2D/3D trajectory GIF frames.",
    )
    parser.add_argument(
        "--max_gif_tracks",
        type=int,
        default=48,
        help="Maximum number of trajectories shown in 2D/3D GIFs.",
    )
    parser.add_argument(
        "--gif_track_sampling",
        type=str,
        default=verify.GIF_TRACK_SAMPLING_SPATIAL,
        choices=verify.GIF_TRACK_SAMPLING_MODES,
        help="How GIF track subsets are chosen when --max_gif_tracks is smaller than the kept track count.",
    )
    parser.add_argument(
        "--max_gif_cloud_points",
        type=int,
        default=3000,
        help="Maximum point-cloud size shown in 3D GIFs.",
    )
    parser.add_argument(
        "--ply_downsample",
        type=int,
        default=4,
        help="Point-cloud downsample factor for the 3D GIF backdrop.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.01,
        help="Minimum valid depth for point-cloud export.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=10.0,
        help="Maximum valid depth for point-cloud export.",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=0.9,
        help="Trajectory alpha for the 2D/3D GIFs.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=1.2,
        help="Trajectory line width for the 2D/3D GIFs.",
    )
    parser.add_argument(
        "--camera_fit_mode",
        type=str,
        default=verify.CAMERA_FIT_SCENE,
        choices=verify.CAMERA_FIT_MODES,
        help="3D panel camera framing mode.",
    )
    parser.add_argument(
        "--trajectory_fit_padding",
        type=float,
        default=1.35,
        help="Extra radius multiplier when --camera_fit_mode=trajectory.",
    )
    parser.add_argument(
        "--panel_height",
        type=int,
        default=DEFAULT_PANEL_HEIGHT,
        help="Output height of each panel inside the composite GIF.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="summary.json",
        help="Summary JSON filename written under output_dir.",
    )
    return parser


def parse_query_frames_from_rgb_gifs(sampled_rgb_gif_dir: Path) -> list[int]:
    query_frames: list[int] = []
    for path in sorted(sampled_rgb_gif_dir.glob("q*_rgb_window.gif")):
        match = RGB_GIF_PATTERN.match(path.name)
        if match is None:
            continue
        query_frames.append(int(match.group(1)))
    if not query_frames:
        raise FileNotFoundError(f"No qXXXXX_rgb_window.gif files found under {sampled_rgb_gif_dir}")
    return sorted(set(query_frames))


def add_label(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, min(image.width, 420), 24), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    return image


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


def load_gif_frames(path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as image:
        frame_idx = 0
        while True:
            try:
                image.seek(frame_idx)
            except EOFError:
                break
            frames.append(image.convert("RGB").copy())
            frame_idx += 1
    if not frames:
        raise ValueError(f"GIF has no frames: {path}")
    return frames


def load_query_bundle(
    *,
    camera_dir: Path,
    query_frame: int,
    render_mode: str,
    ply_downsample: int,
    depth_min: float,
    depth_max: float,
    max_gif_tracks: int,
    gif_track_sampling: str,
) -> dict:
    camera_name = camera_dir.name
    sample_path = camera_dir / "samples" / f"{camera_name}_{query_frame}.npz"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing sample NPZ: {sample_path}")

    sample = normalize_sample_data(sample_path)
    render_view = build_sample_visualization_view(sample, render_mode=render_mode)
    traj = render_view["traj_uvz"]
    traj_2d = render_view["traj_2d"]
    traj_secondary = render_view["traj_uvz_secondary"]
    traj_2d_secondary = render_view["traj_2d_secondary"]
    traj_selection = render_view["traj_uvz_finite"] if render_mode == verify.RENDER_MODE_HYBRID else traj
    segment_frame_indices = np.asarray(render_view["segment_frame_indices"], dtype=np.int32)

    with SceneReader(camera_dir) as scene_reader:
        intrinsics, extrinsics = scene_reader.get_camera_arrays()
        traj_world = traj_uvz_to_world(
            traj,
            intrinsics[query_frame].astype(np.float32),
            extrinsics[query_frame].astype(np.float32),
        )
        traj_world_secondary = traj_uvz_to_world(
            traj_secondary,
            intrinsics[query_frame].astype(np.float32),
            extrinsics[query_frame].astype(np.float32),
        )
        traj_world_selection = traj_uvz_to_world(
            traj_selection,
            intrinsics[query_frame].astype(np.float32),
            extrinsics[query_frame].astype(np.float32),
        )
        aligned_rgb_frames = [
            scene_reader.get_rgb_frame(int(frame_idx))
            for frame_idx in segment_frame_indices.tolist()
        ]

    cloud_points, cloud_colors, rgb_query = verify.create_pointcloud(
        camera_dir=camera_dir,
        camera_name=camera_name,
        frame_idx=query_frame,
        downsample=max(1, ply_downsample),
        depth_min=depth_min,
        depth_max=depth_max,
    )
    gif_candidate_indices = verify.choose_gif_candidate_indices(
        traj_world=traj_world_selection,
        max_tracks=max_gif_tracks,
        gif_track_sampling=gif_track_sampling,
        shared_track_indices=None,
    )
    group_labels = verify.build_track_group_labels(
        traj_pick_place_object_mask=render_view["traj_pick_place_object_mask"],
        traj_manipulator_cluster_id=render_view["traj_manipulator_cluster_id"],
        track_indices=gif_candidate_indices,
    )
    gif_track_indices = verify.choose_gif_track_indices(
        track_indices=gif_candidate_indices,
        max_gif_tracks=min(max_gif_tracks, len(gif_candidate_indices)),
        gif_track_sampling=gif_track_sampling,
        group_labels=group_labels,
        query_points=np.asarray(traj_2d[gif_candidate_indices, 0], dtype=np.float32),
    )

    return {
        "camera_name": camera_name,
        "query_frame": int(query_frame),
        "segment_frame_indices": segment_frame_indices,
        "aligned_rgb_frames": aligned_rgb_frames,
        "rgb_query": rgb_query,
        "traj_2d": traj_2d,
        "traj_2d_secondary": traj_2d_secondary,
        "traj_world": traj_world,
        "traj_world_secondary": traj_world_secondary,
        "cloud_points": cloud_points,
        "cloud_colors": cloud_colors,
        "gif_track_indices": gif_track_indices,
    }


def render_aligned_rgb_gif(
    *,
    aligned_rgb_frames: list[np.ndarray],
    segment_frame_indices: np.ndarray,
    query_frame: int,
    output_path: Path,
    fps: int,
) -> int:
    frames: list[Image.Image] = []
    for frame_idx, rgb in zip(segment_frame_indices.tolist(), aligned_rgb_frames):
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        image = add_label(image, f"RGB frame={frame_idx:05d} query={query_frame:05d}")
        frames.append(image)
    save_gif(output_path, frames, fps)
    return len(frames)


def compose_triptych_gif(
    *,
    rgb_gif_path: Path,
    gif_2d_path: Path,
    gif_3d_path: Path,
    output_path: Path,
    fps: int,
    panel_height: int,
    query_frame: int,
) -> int:
    rgb_frames = load_gif_frames(rgb_gif_path)
    frames_2d = load_gif_frames(gif_2d_path)
    frames_3d = load_gif_frames(gif_3d_path)

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

    save_gif(output_path, composite_frames, fps)
    return frame_count


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    camera_dir = args.camera_dir.resolve()
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Camera directory does not exist: {camera_dir}")

    sampled_rgb_gif_dir = args.sampled_rgb_gif_dir.resolve()
    if not sampled_rgb_gif_dir.is_dir():
        raise FileNotFoundError(f"sampled_rgb_gif_dir does not exist: {sampled_rgb_gif_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_rgb_dir = output_dir / "aligned_rgb_gifs"
    tracks_2d_dir = output_dir / "tracks_2d_gifs"
    tracks_3d_dir = output_dir / "tracks_3d_gifs"
    composite_dir = output_dir / "composite_gifs"
    aligned_rgb_dir.mkdir(parents=True, exist_ok=True)
    tracks_2d_dir.mkdir(parents=True, exist_ok=True)
    tracks_3d_dir.mkdir(parents=True, exist_ok=True)
    composite_dir.mkdir(parents=True, exist_ok=True)

    query_frames = parse_query_frames_from_rgb_gifs(sampled_rgb_gif_dir)
    artifacts: list[QueryCompositeArtifact] = []

    for query_frame in query_frames:
        bundle = load_query_bundle(
            camera_dir=camera_dir,
            query_frame=query_frame,
            render_mode=str(args.render_mode),
            ply_downsample=max(1, int(args.ply_downsample)),
            depth_min=float(args.depth_min),
            depth_max=float(args.depth_max),
            max_gif_tracks=max(1, int(args.max_gif_tracks)),
            gif_track_sampling=str(args.gif_track_sampling),
        )
        rgb_gif_path = aligned_rgb_dir / f"q{query_frame:05d}_rgb_segment.gif"
        gif_2d_path = tracks_2d_dir / f"q{query_frame:05d}_2d_tracks.gif"
        gif_3d_path = tracks_3d_dir / f"q{query_frame:05d}_3d_tracks.gif"
        composite_gif_path = composite_dir / f"q{query_frame:05d}_rgb_2d_3d.gif"

        render_aligned_rgb_gif(
            aligned_rgb_frames=bundle["aligned_rgb_frames"],
            segment_frame_indices=bundle["segment_frame_indices"],
            query_frame=query_frame,
            output_path=rgb_gif_path,
            fps=max(1, int(args.gif_fps)),
        )
        verify.create_2d_gif(
            camera_name=bundle["camera_name"],
            query_frame=query_frame,
            rgb=bundle["rgb_query"],
            traj_2d_primary=bundle["traj_2d"],
            traj_2d_secondary=bundle["traj_2d_secondary"],
            track_indices=bundle["gif_track_indices"],
            gif_path=gif_2d_path,
            line_alpha=float(args.line_alpha),
            line_width=float(args.line_width),
            gif_fps=max(1, int(args.gif_fps)),
            gif_dpi=max(60, int(args.gif_dpi)),
        )
        verify.create_3d_gif(
            camera_name=bundle["camera_name"],
            query_frame=query_frame,
            cloud_points=bundle["cloud_points"],
            cloud_colors=bundle["cloud_colors"],
            traj_world_primary=bundle["traj_world"],
            traj_world_secondary=bundle["traj_world_secondary"],
            track_indices=bundle["gif_track_indices"],
            gif_path=gif_3d_path,
            max_cloud_points=max(1, int(args.max_gif_cloud_points)),
            line_alpha=float(args.line_alpha),
            line_width=float(args.line_width),
            gif_fps=max(1, int(args.gif_fps)),
            gif_dpi=max(60, int(args.gif_dpi)),
            camera_fit_mode=str(args.camera_fit_mode),
            trajectory_fit_padding=max(1.0, float(args.trajectory_fit_padding)),
        )
        frame_count = compose_triptych_gif(
            rgb_gif_path=rgb_gif_path,
            gif_2d_path=gif_2d_path,
            gif_3d_path=gif_3d_path,
            output_path=composite_gif_path,
            fps=max(1, int(args.gif_fps)),
            panel_height=max(120, int(args.panel_height)),
            query_frame=query_frame,
        )
        artifact = QueryCompositeArtifact(
            query_frame=int(query_frame),
            segment_frame_indices=bundle["segment_frame_indices"].astype(int).tolist(),
            rgb_gif_path=str(rgb_gif_path),
            tracks_2d_gif_path=str(gif_2d_path),
            tracks_3d_gif_path=str(gif_3d_path),
            composite_gif_path=str(composite_gif_path),
            frame_count=int(frame_count),
        )
        artifacts.append(artifact)
        print(
            f"[query={query_frame:05d}] rgb={rgb_gif_path} "
            f"gif2d={gif_2d_path} gif3d={gif_3d_path} composite={composite_gif_path}"
        )

    summary = {
        "camera_dir": str(camera_dir),
        "sampled_rgb_gif_dir": str(sampled_rgb_gif_dir),
        "output_dir": str(output_dir),
        "query_frames": query_frames,
        "render_mode": str(args.render_mode),
        "gif_fps": int(args.gif_fps),
        "gif_dpi": int(args.gif_dpi),
        "gif_track_sampling": str(args.gif_track_sampling),
        "camera_fit_mode": str(args.camera_fit_mode),
        "trajectory_fit_padding": max(1.0, float(args.trajectory_fit_padding)),
        "panel_height": int(args.panel_height),
        "artifacts": [asdict(item) for item in artifacts],
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
