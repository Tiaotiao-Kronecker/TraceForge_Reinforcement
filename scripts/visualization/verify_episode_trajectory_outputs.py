#!/usr/bin/env python3
"""
Generate per-camera verification artifacts for one episode trajectory result.

Outputs for each selected camera:
1. A binary PLY point cloud for the chosen query frame.
2. A static PNG with 2D trajectory overlay and a 3D trajectory/pointcloud view.
3. Optional 2D/3D GIFs when --export_gifs is enabled.

This is a headless verification script. It does not start a viser server.

Example:
    python scripts/visualization/verify_episode_trajectory_outputs.py \
        --episode_dir /data1/yaoxuran/press_one_button_demo_v2/episode_00018_green \
        --camera_names varied_camera_1,varied_camera_2,varied_camera_3 \
        --query_frames 0,15,30
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


DEFAULT_CAMERAS = [
    "varied_camera_1",
    "varied_camera_2",
    "varied_camera_3",
]

RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


@dataclass(frozen=True)
class CameraArtifact:
    camera_name: str
    query_frame: int
    ply_path: str
    figure_path: str
    gif_2d_path: str | None
    gif_3d_path: str | None
    available_query_frames: list[int]
    exported_point_count: int
    visualized_track_count: int
    gif_track_count: int
    gif_cloud_point_count: int
    animation_frame_count: int


def parse_csv_items(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_query_frames(raw: str, num_cameras: int) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--query_frames must contain at least one integer.")
    if len(values) == 1 and num_cameras > 1:
        return values * num_cameras
    if len(values) != num_cameras:
        raise ValueError(
            f"--query_frames length ({len(values)}) must match camera count ({num_cameras})."
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify one episode trajectory output by exporting PLYs and trajectory figures."
    )
    parser.add_argument(
        "--episode_dir",
        type=Path,
        required=True,
        help="Episode directory that contains trajectory/<camera>/...",
    )
    parser.add_argument(
        "--trajectory_dirname",
        type=str,
        default="trajectory",
        help="Trajectory result directory name under the episode.",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera names.",
    )
    parser.add_argument(
        "--query_frames",
        type=str,
        default="0,15,30",
        help="Comma-separated query frame indices. One value per camera, or one value reused for all cameras.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output root. Defaults to <episode_dir>/_trajectory_verification.",
    )
    parser.add_argument(
        "--ply_downsample",
        type=int,
        default=4,
        help="Point-cloud pixel downsample factor.",
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
        "--max_cloud_points",
        type=int,
        default=20000,
        help="Maximum point count rendered in the 3D PNG.",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=256,
        help="Maximum trajectory count rendered in each verification PNG.",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=0.9,
        help="Alpha used for trajectory lines.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=1.2,
        help="Line width used for trajectory lines.",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=8,
        help="GIF playback FPS when --export_gifs is enabled.",
    )
    parser.add_argument(
        "--gif_dpi",
        type=int,
        default=110,
        help="Raster DPI used for GIF frames when --export_gifs is enabled.",
    )
    parser.add_argument(
        "--max_gif_tracks",
        type=int,
        default=64,
        help="Maximum trajectory count rendered in each GIF when --export_gifs is enabled.",
    )
    parser.add_argument(
        "--max_gif_cloud_points",
        type=int,
        default=4000,
        help="Maximum point count rendered in the 3D GIF when --export_gifs is enabled.",
    )
    parser.add_argument(
        "--export_gifs",
        action="store_true",
        help="Also export per-camera 2D/3D GIFs. Disabled by default because 3D GIF generation is slow.",
    )
    return parser


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_rgb_if_needed(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = target_hw
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb
    resized = Image.fromarray(rgb).resize((width, height), RESAMPLE_LANCZOS)
    return np.array(resized, dtype=np.uint8)


def list_available_query_frames(camera_dir: Path, camera_name: str) -> list[int]:
    return list_sample_query_frames(camera_dir, camera_name)


def resolve_traj_valid_mask(
    data: np.lib.npyio.NpzFile,
    num_tracks: int,
    sample_path: Path,
) -> np.ndarray:
    if "traj_valid_mask" not in data:
        return np.ones(num_tracks, dtype=bool)

    traj_valid_mask = np.asarray(data["traj_valid_mask"]).astype(bool, copy=False)
    if traj_valid_mask.shape != (num_tracks,):
        print(
            f"[warn] ignoring malformed traj_valid_mask in {sample_path}: "
            f"expected {(num_tracks,)}, got {traj_valid_mask.shape}",
            file=sys.stderr,
        )
        return np.ones(num_tracks, dtype=bool)
    return traj_valid_mask


def load_sample_npz(sample_path: Path) -> dict:
    return normalize_sample_data(sample_path)


def save_ply_binary(output_path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors_u8[:, 0]
    vertex["green"] = colors_u8[:, 1]
    vertex["blue"] = colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with output_path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def create_pointcloud(
    *,
    camera_dir: Path,
    camera_name: str,
    frame_idx: int,
    downsample: int,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with SceneReader(camera_dir) as scene_reader:
        intrinsics, extrinsics = scene_reader.get_camera_arrays()
        if frame_idx >= len(intrinsics) or frame_idx >= len(extrinsics):
            raise IndexError(
                f"Frame {frame_idx} exceeds intrinsics/extrinsics length for {camera_name}."
            )
        depth = scene_reader.get_depth_frame(frame_idx)
        rgb = resize_rgb_if_needed(scene_reader.get_rgb_frame(frame_idx), depth.shape)
        points, colors = build_pointcloud_from_frame(
            depth=depth,
            rgb=rgb,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics[frame_idx],
            downsample=downsample,
            depth_min=depth_min,
            depth_max=depth_max,
        )
    return points, colors, rgb


def choose_track_indices(traj_world: np.ndarray, max_tracks: int) -> np.ndarray:
    valid = np.isfinite(traj_world).all(axis=-1)
    seg = np.diff(traj_world, axis=1)
    valid_seg = valid[:, :-1] & valid[:, 1:]
    motion = np.where(valid_seg, np.linalg.norm(seg, axis=-1), 0.0).sum(axis=1)
    nonzero = np.flatnonzero(motion > 0)
    if len(nonzero) == 0:
        nonzero = np.arange(len(traj_world))
    ranked = nonzero[np.argsort(-motion[nonzero])]
    return ranked[: min(max_tracks, len(ranked))]


def make_track_colors(num_tracks: int) -> np.ndarray:
    if num_tracks <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    cmap = plt.colormaps["turbo"]
    values = np.linspace(0.0, 1.0, num_tracks, dtype=np.float32)
    return np.asarray([cmap(float(v))[:3] for v in values], dtype=np.float32)


def set_axes_equal(ax, points: np.ndarray) -> None:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(float(radius), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def figure_to_image(fig: plt.Figure, dpi: int) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB").copy()
    buffer.close()
    return image


def save_gif(output_path: Path, frames: list[Image.Image], fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(fps, 1)), 1)
    paletted = [
        frame.convert("P", palette=Image.ADAPTIVE, colors=256)
        for frame in frames
    ]
    paletted[0].save(
        output_path,
        save_all=True,
        append_images=paletted[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def build_axis_limits(points: np.ndarray) -> tuple[np.ndarray, float]:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(0.5 * np.max(maxs - mins)), 1e-3)
    return center.astype(np.float32), radius


def apply_axis_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def sample_cloud_points(
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    max_cloud_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(cloud_points) <= max_cloud_points:
        return cloud_points, cloud_colors

    rng = np.random.default_rng(0)
    cloud_sel = rng.choice(len(cloud_points), max_cloud_points, replace=False)
    return cloud_points[cloud_sel], cloud_colors[cloud_sel]


def create_2d_gif(
    *,
    camera_name: str,
    query_frame: int,
    rgb: np.ndarray,
    traj_2d: np.ndarray,
    track_indices: np.ndarray,
    gif_path: Path,
    line_alpha: float,
    line_width: float,
    gif_fps: int,
    gif_dpi: int,
) -> int:
    selected_traj_2d = traj_2d[track_indices]
    track_colors = make_track_colors(len(track_indices))
    num_frames = selected_traj_2d.shape[1]
    frames: list[Image.Image] = []

    for t in range(num_frames):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.imshow(rgb)
        for idx, color in enumerate(track_colors):
            traj = selected_traj_2d[idx]
            prefix = traj[: t + 1]
            valid = np.isfinite(prefix).all(axis=1)
            if np.count_nonzero(valid) == 0:
                continue
            pts = prefix[valid]
            if len(pts) >= 2:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=color,
                    linewidth=line_width,
                    alpha=line_alpha,
                )
            ax.scatter(
                pts[-1, 0],
                pts[-1, 1],
                s=14,
                color=color,
                edgecolors="white",
                linewidths=0.3,
            )
        ax.set_title(f"{camera_name} | query frame {query_frame} | t={t} | 2D trajectories")
        ax.set_axis_off()
        frames.append(figure_to_image(fig, dpi=gif_dpi))

    save_gif(gif_path, frames, gif_fps)
    return num_frames


def create_3d_gif(
    *,
    camera_name: str,
    query_frame: int,
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    traj_world: np.ndarray,
    track_indices: np.ndarray,
    gif_path: Path,
    max_cloud_points: int,
    line_alpha: float,
    line_width: float,
    gif_fps: int,
    gif_dpi: int,
) -> int:
    selected_traj_world = traj_world[track_indices]
    track_colors = make_track_colors(len(track_indices))
    num_frames = selected_traj_world.shape[1]
    cloud_points_plot, cloud_colors_plot = sample_cloud_points(
        cloud_points,
        cloud_colors,
        max_cloud_points,
    )

    traj_points = selected_traj_world.reshape(-1, 3)
    center, radius = build_axis_limits(
        np.concatenate([cloud_points_plot, traj_points], axis=0)
    )
    frames: list[Image.Image] = []

    for t in range(num_frames):
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
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
        for idx, color in enumerate(track_colors):
            traj = selected_traj_world[idx]
            prefix = traj[: t + 1]
            valid = np.isfinite(prefix).all(axis=1)
            if np.count_nonzero(valid) == 0:
                continue
            pts = prefix[valid]
            if len(pts) >= 2:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=color,
                    linewidth=line_width,
                    alpha=line_alpha,
                )
            ax.scatter(
                pts[-1, 0],
                pts[-1, 1],
                pts[-1, 2],
                color=color,
                s=10,
                depthshade=False,
            )
        apply_axis_limits(ax, center, radius)
        ax.set_title(f"{camera_name} | query frame {query_frame} | t={t} | world trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-58)
        frames.append(figure_to_image(fig, dpi=gif_dpi))

    save_gif(gif_path, frames, gif_fps)
    return num_frames


def create_verification_figure(
    *,
    camera_name: str,
    query_frame: int,
    rgb: np.ndarray,
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    traj_2d: np.ndarray,
    traj_world: np.ndarray,
    track_indices: np.ndarray,
    figure_path: Path,
    max_cloud_points: int,
    line_alpha: float,
    line_width: float,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    selected_traj_2d = traj_2d[track_indices]
    selected_traj_world = traj_world[track_indices]
    track_colors = make_track_colors(len(track_indices))

    cloud_points_plot, cloud_colors_plot = sample_cloud_points(
        cloud_points,
        cloud_colors,
        max_cloud_points,
    )

    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    ax_img = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    ax_img.imshow(rgb)
    for idx, color in enumerate(track_colors):
        traj = selected_traj_2d[idx]
        valid = np.isfinite(traj).all(axis=1)
        if np.count_nonzero(valid) < 2:
            continue
        pts = traj[valid]
        ax_img.plot(pts[:, 0], pts[:, 1], color=color, linewidth=line_width, alpha=line_alpha)
        ax_img.scatter(
            pts[0, 0],
            pts[0, 1],
            s=10,
            color=color,
            edgecolors="white",
            linewidths=0.3,
        )
    ax_img.set_title(f"{camera_name} | query frame {query_frame} | 2D trajectories")
    ax_img.set_axis_off()

    ax_3d.scatter(
        cloud_points_plot[:, 0],
        cloud_points_plot[:, 1],
        cloud_points_plot[:, 2],
        c=cloud_colors_plot,
        s=0.4,
        alpha=0.8,
        linewidths=0.0,
    )
    line_points: list[np.ndarray] = []
    for idx, color in enumerate(track_colors):
        traj = selected_traj_world[idx]
        valid = np.isfinite(traj).all(axis=1)
        if np.count_nonzero(valid) < 2:
            continue
        pts = traj[valid]
        ax_3d.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            linewidth=line_width,
            alpha=line_alpha,
        )
        ax_3d.scatter(
            pts[0, 0],
            pts[0, 1],
            pts[0, 2],
            color=color,
            s=8,
            depthshade=False,
        )
        line_points.append(pts)
    ax_3d.set_title(f"{camera_name} | query frame {query_frame} | world trajectories")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    all_points = [cloud_points_plot]
    for pts in line_points:
        all_points.append(pts)
    set_axes_equal(ax_3d, np.concatenate(all_points, axis=0))
    ax_3d.view_init(elev=22, azim=-58)

    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def compose_overview(image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    max_width = max(image.width for image in images)
    total_height = sum(image.height for image in images)
    canvas = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    for image in images:
        image.close()


def verify_camera(
    *,
    episode_dir: Path,
    trajectory_dirname: str,
    camera_name: str,
    query_frame: int,
    output_root: Path,
    ply_downsample: int,
    depth_min: float,
    depth_max: float,
    max_cloud_points: int,
    max_tracks: int,
    line_alpha: float,
    line_width: float,
    gif_fps: int,
    gif_dpi: int,
    max_gif_tracks: int,
    max_gif_cloud_points: int,
    export_gifs: bool,
) -> CameraArtifact:
    camera_dir = episode_dir / trajectory_dirname / camera_name
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Missing camera result directory: {camera_dir}")

    available_query_frames = list_available_query_frames(camera_dir, camera_name)
    if query_frame not in available_query_frames:
        raise ValueError(
            f"{camera_name} query frame {query_frame} not found. "
            f"Available frames: {available_query_frames}"
        )

    sample_path = camera_dir / "samples" / f"{camera_name}_{query_frame}.npz"
    sample = load_sample_npz(sample_path)
    traj = sample["traj_uvz"].astype(np.float32)
    traj_2d = sample["traj_2d"].astype(np.float32)
    traj_valid_mask = sample["traj_valid_mask"].astype(bool, copy=False)
    segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32)
    if sample.get("frame_aligned", False) and len(segment_frame_indices) < traj.shape[1]:
        traj = traj[:, : len(segment_frame_indices)]
        traj_2d = traj_2d[:, : len(segment_frame_indices)]

    raw_num_tracks = traj.shape[0]
    if not np.all(traj_valid_mask):
        traj = traj[traj_valid_mask]
        traj_2d = traj_2d[traj_valid_mask]
    with SceneReader(camera_dir) as scene_reader:
        intrinsics, extrinsics = scene_reader.get_camera_arrays()
        if query_frame >= len(intrinsics) or query_frame >= len(extrinsics):
            raise IndexError(
                f"Frame {query_frame} exceeds intrinsics/extrinsics length for {camera_name}."
            )
        traj_world = traj_uvz_to_world(
            traj,
            intrinsics[query_frame].astype(np.float32),
            extrinsics[query_frame].astype(np.float32),
        )
    print(
        f"[{camera_name}] traj_valid_mask kept {len(traj_world)}/{raw_num_tracks} tracks",
        file=sys.stderr,
    )

    cloud_points, cloud_colors, rgb = create_pointcloud(
        camera_dir=camera_dir,
        camera_name=camera_name,
        frame_idx=query_frame,
        downsample=ply_downsample,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    track_indices = choose_track_indices(traj_world, max_tracks)
    gif_track_indices = track_indices[: min(max_gif_tracks, len(track_indices))]
    gif_cloud_point_count_value = int(min(len(cloud_points), max_gif_cloud_points))

    camera_output_dir = output_root / camera_name
    ply_path = camera_output_dir / f"{camera_name}_frame{query_frame:05d}.ply"
    figure_path = camera_output_dir / f"{camera_name}_frame{query_frame:05d}_verification.png"
    gif_2d_path = camera_output_dir / f"{camera_name}_frame{query_frame:05d}_2d_tracks.gif"
    gif_3d_path = camera_output_dir / f"{camera_name}_frame{query_frame:05d}_3d_tracks.gif"

    save_ply_binary(ply_path, cloud_points, cloud_colors)
    create_verification_figure(
        camera_name=camera_name,
        query_frame=query_frame,
        rgb=rgb,
        cloud_points=cloud_points,
        cloud_colors=cloud_colors,
        traj_2d=traj_2d,
        traj_world=traj_world,
        track_indices=track_indices,
        figure_path=figure_path,
        max_cloud_points=max_cloud_points,
        line_alpha=line_alpha,
        line_width=line_width,
    )
    gif_2d_path_value: str | None = None
    gif_3d_path_value: str | None = None
    gif_track_count = 0
    gif_cloud_point_count = 0
    frame_count_2d = 0
    frame_count_3d = 0
    if export_gifs:
        frame_count_2d = create_2d_gif(
            camera_name=camera_name,
            query_frame=query_frame,
            rgb=rgb,
            traj_2d=traj_2d,
            track_indices=gif_track_indices,
            gif_path=gif_2d_path,
            line_alpha=line_alpha,
            line_width=line_width,
            gif_fps=gif_fps,
            gif_dpi=gif_dpi,
        )
        frame_count_3d = create_3d_gif(
            camera_name=camera_name,
            query_frame=query_frame,
            cloud_points=cloud_points,
            cloud_colors=cloud_colors,
            traj_world=traj_world,
            track_indices=gif_track_indices,
            gif_path=gif_3d_path,
            max_cloud_points=max_gif_cloud_points,
            line_alpha=line_alpha,
            line_width=line_width,
            gif_fps=gif_fps,
            gif_dpi=gif_dpi,
        )
        gif_2d_path_value = str(gif_2d_path)
        gif_3d_path_value = str(gif_3d_path)
        gif_track_count = int(len(gif_track_indices))
        gif_cloud_point_count = gif_cloud_point_count_value

    return CameraArtifact(
        camera_name=camera_name,
        query_frame=query_frame,
        ply_path=str(ply_path),
        figure_path=str(figure_path),
        gif_2d_path=gif_2d_path_value,
        gif_3d_path=gif_3d_path_value,
        available_query_frames=available_query_frames,
        exported_point_count=int(len(cloud_points)),
        visualized_track_count=int(len(track_indices)),
        gif_track_count=gif_track_count,
        gif_cloud_point_count=gif_cloud_point_count,
        animation_frame_count=int(min(frame_count_2d, frame_count_3d)),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")

    camera_names = parse_csv_items(args.camera_names)
    query_frames = parse_query_frames(args.query_frames, len(camera_names))
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (episode_dir / "_trajectory_verification").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts: list[CameraArtifact] = []
    for camera_name, query_frame in zip(camera_names, query_frames):
        artifact = verify_camera(
            episode_dir=episode_dir,
            trajectory_dirname=args.trajectory_dirname,
            camera_name=camera_name,
            query_frame=query_frame,
            output_root=output_root,
            ply_downsample=max(1, args.ply_downsample),
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            max_cloud_points=max(1, args.max_cloud_points),
            max_tracks=max(1, args.max_tracks),
            line_alpha=float(args.line_alpha),
            line_width=float(args.line_width),
            gif_fps=max(1, args.gif_fps),
            gif_dpi=max(60, args.gif_dpi),
            max_gif_tracks=max(1, args.max_gif_tracks),
            max_gif_cloud_points=max(1, args.max_gif_cloud_points),
            export_gifs=bool(args.export_gifs),
        )
        artifacts.append(artifact)
        message = (
            f"[{camera_name}] frame={query_frame} "
            f"ply={artifact.ply_path} figure={artifact.figure_path}"
        )
        if artifact.gif_2d_path is not None and artifact.gif_3d_path is not None:
            message += f" gif2d={artifact.gif_2d_path} gif3d={artifact.gif_3d_path}"
        print(message)

    overview_path = output_root / "episode_verification_overview.png"
    compose_overview([Path(item.figure_path) for item in artifacts], overview_path)

    summary = {
        "episode_dir": str(episode_dir),
        "trajectory_dirname": args.trajectory_dirname,
        "output_dir": str(output_root),
        "export_gifs": bool(args.export_gifs),
        "overview_path": str(overview_path),
        "artifacts": [asdict(item) for item in artifacts],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"overview={overview_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
