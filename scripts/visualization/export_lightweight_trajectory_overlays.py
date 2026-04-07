#!/usr/bin/env python3
"""
Export lightweight 2D trajectory overlay artifacts without numpy/matplotlib.

This utility is intended for quick inspection on machines that only have the
Python standard library plus Pillow available. It reads TraceForge sample NPZ
files directly, overlays selected 2D trajectories on the source RGB frames, and
exports:

1. A static query-frame PNG with all selected trajectory trails.
2. A per-query animated GIF over the tracked segment.
3. A summary JSON plus an optional episode-level overview PNG.

Example:
    python3 scripts/visualization/export_lightweight_trajectory_overlays.py \
        --episode_dir /DATA/disk1/zoyo/mcap/wipe_the_table_gs/00000 \
        --camera_names varied_camera_1,varied_camera_2 \
        --query_frames 20 \
        --output_dir /DATA/disk3/tmp/external_preview_20260407/wipe_the_table_gs_00000_q20
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import struct
import zipfile
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_CAMERAS = ["varied_camera_1", "varied_camera_2"]
DEFAULT_QUERY_FRAMES = "0"
DEFAULT_TRAJECTORY_DIRNAME = "trajectory"
DEFAULT_MAX_TRACKS = 96
DEFAULT_TRAIL_STEPS = 12
DEFAULT_GIF_FPS = 10
DEFAULT_POINT_RADIUS = 3
DEFAULT_LINE_WIDTH = 2
DEFAULT_FONT = ImageFont.load_default()
DEFAULT_COLORS = (
    (255, 99, 71),
    (65, 105, 225),
    (60, 179, 113),
    (255, 165, 0),
    (186, 85, 211),
    (0, 191, 255),
    (255, 215, 0),
    (220, 20, 60),
    (72, 209, 204),
    (255, 105, 180),
    (46, 139, 87),
    (123, 104, 238),
)


@dataclass(frozen=True)
class CameraArtifact:
    camera_name: str
    query_frame: int
    sample_path: str
    figure_path: str
    gif_path: str
    available_query_frames: list[int]
    selected_track_count: int
    animation_frame_count: int
    segment_frame_indices: list[int]
    selected_track_indices: list[int]


class NpyArray:
    def __init__(self, *, descr: str, shape: tuple[int, ...], data: bytes):
        self.descr = str(descr)
        self.shape = tuple(int(dim) for dim in shape)
        self.data = data

        if self.descr == "<f4":
            self._format = "<f"
            self.itemsize = 4
        elif self.descr == "<i4":
            self._format = "<i"
            self.itemsize = 4
        elif self.descr == "|b1":
            self._format = None
            self.itemsize = 1
        else:
            raise ValueError(f"Unsupported dtype in lightweight reader: {self.descr}")

        total_items = 1
        for dim in self.shape:
            total_items *= int(dim)
        if len(self.data) != total_items * self.itemsize:
            raise ValueError(
                f"Array byte length mismatch for dtype={self.descr} shape={self.shape}: "
                f"{len(self.data)} vs expected {total_items * self.itemsize}"
            )

        self.total_items = total_items

    def _flat_index(self, indexes: tuple[int, ...]) -> int:
        if len(indexes) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices for shape {self.shape}, got {indexes}")
        stride = 1
        flat_index = 0
        for dim, index in zip(reversed(self.shape), reversed(indexes)):
            index = int(index)
            if index < 0 or index >= dim:
                raise IndexError(f"Index {indexes} out of range for shape {self.shape}")
            flat_index += index * stride
            stride *= dim
        return flat_index

    def _get_scalar(self, flat_index: int) -> Any:
        if flat_index < 0 or flat_index >= self.total_items:
            raise IndexError(flat_index)
        offset = flat_index * self.itemsize
        if self.descr == "|b1":
            return bool(self.data[offset])
        return struct.unpack_from(self._format, self.data, offset)[0]

    def get(self, *indexes: int) -> Any:
        return self._get_scalar(self._flat_index(tuple(indexes)))

    def to_list(self) -> list[Any]:
        return [self._get_scalar(index) for index in range(self.total_items)]


def parse_csv_items(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_query_frames(raw: str, num_cameras: int) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("--query_frames must contain at least one integer.")
    if len(values) == 1 and num_cameras > 1:
        return values * num_cameras
    if len(values) != num_cameras:
        raise ValueError(
            f"--query_frames length ({len(values)}) must match camera count ({num_cameras})."
        )
    return values


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def list_available_query_frames(camera_dir: Path, camera_name: str) -> list[int]:
    sample_dir = camera_dir / "samples"
    frames: list[int] = []
    for sample_path in sorted(sample_dir.glob(f"{camera_name}_*.npz")):
        try:
            frames.append(int(sample_path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(set(frames))


def read_npy_from_zip(zf: zipfile.ZipFile, name: str) -> NpyArray:
    with zf.open(name) as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"Unexpected npy magic for {name}: {magic!r}")
        major, minor = handle.read(2)
        version = bytes([major, minor])
        if version == b"\x01\x00":
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif version in {b"\x02\x00", b"\x03\x00"}:
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"Unsupported npy version for {name}: {version!r}")
        header = ast.literal_eval(handle.read(header_len).decode("latin1"))
        if bool(header.get("fortran_order")):
            raise ValueError(f"Fortran-order arrays are unsupported in lightweight reader: {name}")
        return NpyArray(
            descr=str(header["descr"]),
            shape=tuple(int(dim) for dim in header["shape"]),
            data=handle.read(),
        )


def load_sample_arrays(sample_path: Path) -> dict[str, NpyArray]:
    arrays: dict[str, NpyArray] = {}
    with zipfile.ZipFile(sample_path) as zf:
        for name in (
            "traj_uvz.npy",
            "traj_valid_mask.npy",
            "segment_frame_indices.npy",
            "query_frame_index.npy",
        ):
            arrays[name[:-4]] = read_npy_from_zip(zf, name)
    return arrays


def build_rgb_index(scene_meta: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    source_rgb_dir = Path(str(scene_meta["source_rgb_path"]))
    rgb_map: dict[str, Path] = {}
    for path in sorted(source_rgb_dir.iterdir()):
        if not path.is_file():
            continue
        rgb_map[path.stem] = path
    if not rgb_map:
        raise FileNotFoundError(f"No RGB frames found under {source_rgb_dir}")
    return source_rgb_dir, rgb_map


def resolve_rgb_frame_path(
    *,
    scene_meta: dict[str, Any],
    rgb_map: dict[str, Path],
    local_frame_index: int,
) -> Path:
    original_filenames = scene_meta.get("original_filenames")
    if not isinstance(original_filenames, list):
        raise ValueError("scene_meta original_filenames must be a list")
    if local_frame_index < 0 or local_frame_index >= len(original_filenames):
        raise IndexError(local_frame_index)
    stem = str(original_filenames[local_frame_index])
    frame_path = rgb_map.get(stem)
    if frame_path is None:
        raise FileNotFoundError(f"Missing RGB frame for stem {stem}")
    return frame_path


def compute_track_motion(traj_uvz: NpyArray, track_index: int) -> float:
    num_steps = traj_uvz.shape[1]
    motion = 0.0
    prev_xy: tuple[float, float] | None = None
    for step_index in range(num_steps):
        u = float(traj_uvz.get(track_index, step_index, 0))
        v = float(traj_uvz.get(track_index, step_index, 1))
        z = float(traj_uvz.get(track_index, step_index, 2))
        if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(z)):
            prev_xy = None
            continue
        current_xy = (u, v)
        if prev_xy is not None:
            motion += math.dist(prev_xy, current_xy)
        prev_xy = current_xy
    return motion


def select_tracks(
    *,
    traj_uvz: NpyArray,
    traj_valid_mask: NpyArray,
    max_tracks: int,
) -> list[int]:
    candidates: list[tuple[float, int]] = []
    for track_index in range(traj_uvz.shape[0]):
        if not bool(traj_valid_mask.get(track_index)):
            continue
        motion = compute_track_motion(traj_uvz, track_index)
        if motion <= 0.0:
            continue
        candidates.append((motion, track_index))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [track_index for _, track_index in candidates[: max(1, int(max_tracks))]]
    return selected


def draw_label(image: Image.Image, lines: list[str]) -> None:
    draw = ImageDraw.Draw(image)
    line_height = 14
    box_height = 8 + len(lines) * line_height
    box_width = max(220, max((len(line) for line in lines), default=0) * 7 + 16)
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0, 180))
    y = 4
    for line in lines:
        draw.text((6, y), line, fill=(255, 255, 255), font=DEFAULT_FONT)
        y += line_height


def overlay_trajectories(
    *,
    base_image: Image.Image,
    traj_uvz: NpyArray,
    selected_track_indices: list[int],
    step_limit: int,
    colors: dict[int, tuple[int, int, int]],
    point_radius: int,
    line_width: int,
    trail_steps: int | None,
) -> Image.Image:
    image = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for track_index in selected_track_indices:
        color = colors[track_index]
        points: list[tuple[float, float]] = []
        for step_index in range(step_limit):
            u = float(traj_uvz.get(track_index, step_index, 0))
            v = float(traj_uvz.get(track_index, step_index, 1))
            z = float(traj_uvz.get(track_index, step_index, 2))
            if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(z)):
                continue
            points.append((u, v))
        if trail_steps is not None and trail_steps > 0:
            points = points[-trail_steps:]
        if len(points) >= 2:
            draw.line(points, fill=color, width=max(1, int(line_width)))
        if points:
            px, py = points[-1]
            draw.ellipse(
                (
                    px - point_radius,
                    py - point_radius,
                    px + point_radius,
                    py + point_radius,
                ),
                fill=color,
                outline=(255, 255, 255),
            )
    return image


def save_gif(path: Path, frames: list[Image.Image], fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {path}")
    duration_ms = max(1, int(round(1000.0 / max(1, int(fps)))))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def build_overview_image(camera_rows: list[tuple[str, Path]]) -> Image.Image:
    if not camera_rows:
        raise ValueError("No camera rows available for overview image.")
    images = []
    for camera_name, figure_path in camera_rows:
        image = Image.open(figure_path).convert("RGB")
        draw_label(image, [camera_name])
        images.append(image)
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def export_camera_artifact(
    *,
    episode_dir: Path,
    trajectory_dirname: str,
    camera_name: str,
    query_frame: int,
    output_dir: Path,
    max_tracks: int,
    point_radius: int,
    line_width: int,
    trail_steps: int,
    gif_fps: int,
) -> CameraArtifact:
    camera_dir = episode_dir / trajectory_dirname / camera_name
    sample_path = camera_dir / "samples" / f"{camera_name}_{int(query_frame)}.npz"
    if not sample_path.is_file():
        available_query_frames = list_available_query_frames(camera_dir, camera_name)
        raise FileNotFoundError(
            f"Missing sample NPZ: {sample_path}. Available query frames: {available_query_frames}"
        )

    scene_meta = load_json(camera_dir / "scene_meta.json")
    _, rgb_map = build_rgb_index(scene_meta)
    arrays = load_sample_arrays(sample_path)
    traj_uvz = arrays["traj_uvz"]
    traj_valid_mask = arrays["traj_valid_mask"]
    segment_frame_indices = [int(value) for value in arrays["segment_frame_indices"].to_list()]
    resolved_query_frame = int(arrays["query_frame_index"].get(0))
    selected_track_indices = select_tracks(
        traj_uvz=traj_uvz,
        traj_valid_mask=traj_valid_mask,
        max_tracks=max_tracks,
    )
    if not selected_track_indices:
        raise ValueError(f"No valid moving tracks found in {sample_path}")

    camera_output_dir = output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)
    color_cycle = cycle(DEFAULT_COLORS)
    colors = {track_index: next(color_cycle) for track_index in selected_track_indices}

    rgb_frames: list[Image.Image] = []
    gif_frames: list[Image.Image] = []
    for step_index, local_frame_index in enumerate(segment_frame_indices):
        rgb_path = resolve_rgb_frame_path(
            scene_meta=scene_meta,
            rgb_map=rgb_map,
            local_frame_index=int(local_frame_index),
        )
        rgb_frame = Image.open(rgb_path).convert("RGB")
        rgb_frames.append(rgb_frame)
        gif_frame = overlay_trajectories(
            base_image=rgb_frame,
            traj_uvz=traj_uvz,
            selected_track_indices=selected_track_indices,
            step_limit=step_index + 1,
            colors=colors,
            point_radius=point_radius,
            line_width=line_width,
            trail_steps=trail_steps,
        )
        draw_label(
            gif_frame,
            [
                f"{camera_name} q={resolved_query_frame}",
                f"segment step={step_index + 1}/{len(segment_frame_indices)}",
                f"tracks={len(selected_track_indices)}",
            ],
        )
        gif_frames.append(gif_frame)

    static_image = overlay_trajectories(
        base_image=rgb_frames[0],
        traj_uvz=traj_uvz,
        selected_track_indices=selected_track_indices,
        step_limit=len(segment_frame_indices),
        colors=colors,
        point_radius=point_radius,
        line_width=line_width,
        trail_steps=None,
    )
    draw_label(
        static_image,
        [
            f"{camera_name} q={resolved_query_frame}",
            f"tracks={len(selected_track_indices)}",
            f"steps={len(segment_frame_indices)}",
        ],
    )

    figure_path = camera_output_dir / f"{camera_name}_frame{resolved_query_frame:05d}_overlay.png"
    gif_path = camera_output_dir / f"{camera_name}_frame{resolved_query_frame:05d}_overlay.gif"
    static_image.save(figure_path)
    save_gif(gif_path, gif_frames, gif_fps)

    return CameraArtifact(
        camera_name=camera_name,
        query_frame=resolved_query_frame,
        sample_path=str(sample_path),
        figure_path=str(figure_path),
        gif_path=str(gif_path),
        available_query_frames=list_available_query_frames(camera_dir, camera_name),
        selected_track_count=len(selected_track_indices),
        animation_frame_count=len(gif_frames),
        segment_frame_indices=segment_frame_indices,
        selected_track_indices=selected_track_indices,
    )


def _coerce_int_list(raw: Any, *, field_name: str) -> list[int]:
    if not isinstance(raw, list):
        raise ValueError(f"summary artifact field '{field_name}' must be a list")
    return [int(value) for value in raw]


def load_existing_artifacts(
    *,
    summary_path: Path,
    episode_dir: Path,
    trajectory_dirname: str,
) -> dict[str, CameraArtifact]:
    if not summary_path.is_file():
        return {}

    payload = load_json(summary_path)
    existing_episode_dir = Path(str(payload.get("episode_dir", ""))).resolve()
    existing_trajectory_dirname = str(payload.get("trajectory_dirname", ""))
    if existing_episode_dir != episode_dir or existing_trajectory_dirname != trajectory_dirname:
        raise ValueError(
            "Existing summary.json belongs to a different episode/output layout. "
            f"expected episode_dir={episode_dir} trajectory_dirname={trajectory_dirname}, "
            f"got episode_dir={existing_episode_dir} trajectory_dirname={existing_trajectory_dirname}"
        )

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise ValueError(f"summary.json missing list field 'artifacts': {summary_path}")

    artifacts_by_camera: dict[str, CameraArtifact] = {}
    for item in artifacts_raw:
        if not isinstance(item, dict):
            raise ValueError(f"summary.json artifacts must contain objects: {summary_path}")
        artifact = CameraArtifact(
            camera_name=str(item["camera_name"]),
            query_frame=int(item["query_frame"]),
            sample_path=str(item["sample_path"]),
            figure_path=str(item["figure_path"]),
            gif_path=str(item["gif_path"]),
            available_query_frames=_coerce_int_list(
                item["available_query_frames"],
                field_name="available_query_frames",
            ),
            selected_track_count=int(item["selected_track_count"]),
            animation_frame_count=int(item["animation_frame_count"]),
            segment_frame_indices=_coerce_int_list(
                item["segment_frame_indices"],
                field_name="segment_frame_indices",
            ),
            selected_track_indices=_coerce_int_list(
                item["selected_track_indices"],
                field_name="selected_track_indices",
            ),
        )
        artifacts_by_camera[artifact.camera_name] = artifact
    return artifacts_by_camera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export lightweight 2D trajectory overlays using only Pillow."
    )
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--trajectory_dirname", type=str, default=DEFAULT_TRAJECTORY_DIRNAME)
    parser.add_argument(
        "--camera_names",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera names.",
    )
    parser.add_argument(
        "--query_frames",
        type=str,
        default=DEFAULT_QUERY_FRAMES,
        help="Comma-separated query frame indices. One value may be reused for all cameras.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output root. Defaults to <episode_dir>/_lightweight_trajectory_overlays.",
    )
    parser.add_argument("--max_tracks", type=int, default=DEFAULT_MAX_TRACKS)
    parser.add_argument("--point_radius", type=int, default=DEFAULT_POINT_RADIUS)
    parser.add_argument("--line_width", type=int, default=DEFAULT_LINE_WIDTH)
    parser.add_argument("--trail_steps", type=int, default=DEFAULT_TRAIL_STEPS)
    parser.add_argument("--gif_fps", type=int, default=DEFAULT_GIF_FPS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    camera_names = parse_csv_items(args.camera_names)
    query_frames = parse_query_frames(args.query_frames, len(camera_names))
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (episode_dir / "_lightweight_trajectory_overlays").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    artifacts_by_camera = load_existing_artifacts(
        summary_path=summary_path,
        episode_dir=episode_dir,
        trajectory_dirname=str(args.trajectory_dirname),
    )

    for camera_name, query_frame in zip(camera_names, query_frames):
        artifact = export_camera_artifact(
            episode_dir=episode_dir,
            trajectory_dirname=str(args.trajectory_dirname),
            camera_name=str(camera_name),
            query_frame=int(query_frame),
            output_dir=output_dir,
            max_tracks=int(args.max_tracks),
            point_radius=int(args.point_radius),
            line_width=int(args.line_width),
            trail_steps=int(args.trail_steps),
            gif_fps=int(args.gif_fps),
        )
        artifacts_by_camera[artifact.camera_name] = artifact

    artifacts = [artifacts_by_camera[camera_name] for camera_name in sorted(artifacts_by_camera)]
    overview_rows = [(artifact.camera_name, Path(artifact.figure_path)) for artifact in artifacts]

    overview_path = output_dir / "episode_overlay_overview.png"
    build_overview_image(overview_rows).save(overview_path)

    summary = {
        "episode_dir": str(episode_dir),
        "trajectory_dirname": str(args.trajectory_dirname),
        "output_dir": str(output_dir),
        "overview_path": str(overview_path),
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary_json={summary_path}")
    print(f"overview_png={overview_path}")


if __name__ == "__main__":
    main()
