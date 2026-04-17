#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.query_visibility_gate_utils import (
    compute_query_visibility_gate,
    summarize_query_visibility_gate,
)
from utils.traceforge_artifact_utils import SceneReader, list_sample_query_frames, normalize_sample_data


def _parse_query_frames(raw: str | None, *, episode_dir: Path) -> list[int]:
    if raw is None or not str(raw).strip():
        frames = list_sample_query_frames(episode_dir, episode_dir.name)
        if not frames:
            raise FileNotFoundError(f"No sample NPZ files found under {episode_dir / 'samples'}")
        return frames
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one query frame.")
    return [int(item) for item in values]


def _default_output_dir(episode_dir: Path) -> Path:
    camera_dir = episode_dir.resolve()
    trajectory_dir = camera_dir.parent
    case_dir = trajectory_dir.parent
    return case_dir / "_analysis_query_visibility_gate" / trajectory_dir.name / camera_dir.name


def _save_gif(output_path: Path, frames: list[Image.Image], fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames available for GIF: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(round(1000.0 / max(int(fps), 1))), 1)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _draw_points(
    image: Image.Image,
    *,
    points_xy: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    radius_px: int,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    points_xy = np.asarray(points_xy, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    radius_px = max(int(radius_px), 1)
    for track_idx in np.flatnonzero(mask):
        x = float(points_xy[track_idx, 0])
        y = float(points_xy[track_idx, 1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        draw.ellipse(
            (
                x - radius_px,
                y - radius_px,
                x + radius_px,
                y + radius_px,
            ),
            fill=color_rgb,
            outline=(0, 0, 0),
            width=1,
        )
    return output


def _add_label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    line_count = max(text.count("\n") + 1, 1)
    box_height = 10 + 14 * line_count
    box_width = min(labeled.width, max(420, 7 * max(len(line) for line in text.splitlines())))
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))
    draw.multiline_text((6, 6), text, fill=(255, 255, 255), font=font, spacing=2)
    return labeled


def _render_query_overlay(
    *,
    rgb_frame: np.ndarray,
    keypoints: np.ndarray,
    reliable_mask: np.ndarray,
    output_path: Path,
    label: str,
    point_radius_px: int,
) -> None:
    image = Image.fromarray(np.asarray(rgb_frame, dtype=np.uint8), mode="RGB")
    image = _draw_points(
        image,
        points_xy=keypoints,
        mask=~np.asarray(reliable_mask, dtype=bool),
        color_rgb=(230, 57, 70),
        radius_px=point_radius_px,
    )
    image = _draw_points(
        image,
        points_xy=keypoints,
        mask=np.asarray(reliable_mask, dtype=bool),
        color_rgb=(46, 204, 113),
        radius_px=point_radius_px,
    )
    image = _add_label(image, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _render_projection_gif(
    *,
    rgb_frames: list[np.ndarray],
    frame_indices: np.ndarray,
    projected_uvz: np.ndarray,
    projected_in_bounds_mask: np.ndarray,
    reliable_mask: np.ndarray,
    query_frame: int,
    output_path: Path,
    fps: int,
    point_radius_px: int,
) -> None:
    projected_uvz = np.asarray(projected_uvz, dtype=np.float32)
    projected_in_bounds_mask = np.asarray(projected_in_bounds_mask, dtype=bool)
    reliable_mask = np.asarray(reliable_mask, dtype=bool)

    frames: list[Image.Image] = []
    for local_frame_idx, (rgb_frame, source_frame_idx) in enumerate(
        zip(rgb_frames, frame_indices.tolist(), strict=False)
    ):
        image = Image.fromarray(np.asarray(rgb_frame, dtype=np.uint8), mode="RGB")
        visible_mask = projected_in_bounds_mask[:, local_frame_idx]
        image = _draw_points(
            image,
            points_xy=projected_uvz[:, local_frame_idx, :2],
            mask=visible_mask & (~reliable_mask),
            color_rgb=(230, 57, 70),
            radius_px=point_radius_px,
        )
        image = _draw_points(
            image,
            points_xy=projected_uvz[:, local_frame_idx, :2],
            mask=visible_mask & reliable_mask,
            color_rgb=(46, 204, 113),
            radius_px=point_radius_px,
        )
        label = (
            f"query={query_frame} source_frame={source_frame_idx}\n"
            f"green=keep({int(np.count_nonzero(visible_mask & reliable_mask))}) "
            f"red=remove({int(np.count_nonzero(visible_mask & (~reliable_mask)))})"
        )
        frames.append(_add_label(image, label))
    _save_gif(output_path, frames, fps=fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize first-frame 3D query-point visibility gating over 2D RGB frames."
    )
    parser.add_argument("--episode_dir", type=Path, required=True, help="Trajectory camera directory with samples/.")
    parser.add_argument("--query_frames", type=str, default=None, help="Comma-separated query frames. Defaults to all.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_border_dist_px", type=float, default=0.0)
    parser.add_argument("--gif_fps", type=int, default=6)
    parser.add_argument("--point_radius_px", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_dir = args.episode_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir(episode_dir)
    query_frames = _parse_query_frames(args.query_frames, episode_dir=episode_dir)

    summaries: list[dict[str, object]] = []
    with SceneReader(episode_dir) as scene_reader:
        intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
        for query_frame in query_frames:
            sample_path = episode_dir / "samples" / f"{episode_dir.name}_{int(query_frame)}.npz"
            sample = normalize_sample_data(sample_path)
            segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32).reshape(-1)
            if segment_frame_indices.size == 0:
                raise ValueError(f"Sample has empty segment_frame_indices: {sample_path}")
            if int(segment_frame_indices[0]) != int(query_frame):
                raise ValueError(
                    f"Expected sample to start at query frame {query_frame}, got {int(segment_frame_indices[0])}"
                )

            depth_segment = np.stack(
                [scene_reader.get_depth_frame(int(frame_idx)).astype(np.float32) for frame_idx in segment_frame_indices],
                axis=0,
            ).astype(np.float32)
            rgb_frames = [
                scene_reader.get_rgb_frame(int(frame_idx)).astype(np.uint8)
                for frame_idx in segment_frame_indices.tolist()
            ]
            intrinsics_segment = intrinsics_all[segment_frame_indices].astype(np.float32, copy=False)
            extrinsics_segment = extrinsics_all[segment_frame_indices].astype(np.float32, copy=False)

            gate_result = compute_query_visibility_gate(
                depth_segment,
                intrinsics_segment,
                extrinsics_segment,
                keypoints=np.asarray(sample["keypoints"], dtype=np.float32),
                query_frame=0,
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                min_border_dist_px=float(args.min_border_dist_px),
            )
            summary = summarize_query_visibility_gate(
                gate_result=gate_result,
                traj_valid_mask=np.asarray(sample["traj_valid_mask"], dtype=bool),
            )
            summary_payload = {
                "episode_dir": str(episode_dir),
                "sample_npz": str(sample_path),
                "query_frame": int(query_frame),
                "segment_frame_indices": segment_frame_indices.astype(int).tolist(),
                "gate_summary": gate_result["summary"],
                "tracked_summary": summary,
            }

            query_output_dir = output_dir / f"q{int(query_frame):05d}"
            query_output_dir.mkdir(parents=True, exist_ok=True)
            (query_output_dir / "summary.json").write_text(
                json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n"
            )

            reliable_mask = np.asarray(gate_result["reliable_track_mask"], dtype=bool)
            label = (
                f"query={query_frame}\n"
                f"green=keep({int(np.count_nonzero(reliable_mask))}) "
                f"red=remove({int(np.count_nonzero(~reliable_mask))})"
            )
            _render_query_overlay(
                rgb_frame=rgb_frames[0],
                keypoints=np.asarray(sample["keypoints"], dtype=np.float32),
                reliable_mask=reliable_mask,
                output_path=query_output_dir / "query_overlay.png",
                label=label,
                point_radius_px=int(args.point_radius_px),
            )
            _render_projection_gif(
                rgb_frames=rgb_frames,
                frame_indices=segment_frame_indices,
                projected_uvz=np.asarray(gate_result["projected_uvz"], dtype=np.float32),
                projected_in_bounds_mask=np.asarray(gate_result["projected_in_bounds_mask"], dtype=bool),
                reliable_mask=reliable_mask,
                query_frame=int(query_frame),
                output_path=query_output_dir / "projected_visibility.gif",
                fps=int(args.gif_fps),
                point_radius_px=int(args.point_radius_px),
            )
            summaries.append(summary_payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
