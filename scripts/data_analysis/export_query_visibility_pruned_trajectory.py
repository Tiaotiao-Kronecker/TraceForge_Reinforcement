#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

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


def _default_output_dir(episode_dir: Path, *, track_mask_mode: str) -> Path:
    camera_dir = episode_dir.resolve()
    trajectory_dir = camera_dir.parent
    if track_mask_mode == "kept":
        suffix = "vispruneq3d"
    elif track_mask_mode == "removed":
        suffix = "visremovedq3d"
    else:
        raise ValueError(f"Unsupported track_mask_mode: {track_mask_mode}")
    return trajectory_dir.with_name(f"{trajectory_dir.name}_{suffix}") / camera_dir.name


def _copy_required_metadata(episode_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_meta_src = episode_dir / "scene_meta.json"
    scene_meta_dst = output_dir / "scene_meta.json"
    if not scene_meta_src.is_file():
        raise FileNotFoundError(f"Missing scene_meta.json under {episode_dir}")
    shutil.copy2(scene_meta_src, scene_meta_dst)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a post-track trajectory copy selected by first-frame 3D visibility gate."
    )
    parser.add_argument("--episode_dir", type=Path, required=True, help="Trajectory camera directory with samples/.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--query_frames", type=str, default=None, help="Comma-separated query frames. Defaults to all.")
    parser.add_argument(
        "--track_mask_mode",
        type=str,
        default="kept",
        choices=("kept", "removed"),
        help="Whether the exported traj_valid_mask keeps reliable tracks or only the removed tracks.",
    )
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_border_dist_px", type=float, default=0.0)
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Optional explicit summary JSON path. Defaults to <output_dir>/query_visibility_prune_summary.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_dir = args.episode_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_dir(episode_dir, track_mask_mode=str(args.track_mask_mode))
    )
    summary_json = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else (output_dir / "query_visibility_prune_summary.json").resolve()
    )
    query_frames = set(_parse_query_frames(args.query_frames, episode_dir=episode_dir))

    _copy_required_metadata(episode_dir, output_dir)
    samples_src_dir = episode_dir / "samples"
    samples_dst_dir = output_dir / "samples"
    samples_dst_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    with SceneReader(episode_dir) as scene_reader:
        intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
        for sample_path in sorted(samples_src_dir.glob("*.npz")):
            sample = normalize_sample_data(sample_path)
            query_frame = int(sample["query_frame_index"])
            raw_payload = dict(np.load(sample_path, allow_pickle=False))
            dst_path = samples_dst_dir / sample_path.name

            if query_frame not in query_frames:
                np.savez(dst_path, **raw_payload)
                continue

            segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32).reshape(-1)
            depth_segment = np.stack(
                [scene_reader.get_depth_frame(int(frame_idx)).astype(np.float32) for frame_idx in segment_frame_indices],
                axis=0,
            ).astype(np.float32)
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
            gate_summary = summarize_query_visibility_gate(
                gate_result=gate_result,
                traj_valid_mask=np.asarray(sample["traj_valid_mask"], dtype=bool),
            )

            original_valid_mask = np.asarray(raw_payload["traj_valid_mask"], dtype=bool).reshape(-1)
            reliable_track_mask = np.asarray(gate_result["reliable_track_mask"], dtype=bool).reshape(-1)
            if str(args.track_mask_mode) == "kept":
                selected_valid_mask = original_valid_mask & reliable_track_mask
            else:
                selected_valid_mask = original_valid_mask & (~reliable_track_mask)

            raw_payload["traj_valid_mask"] = selected_valid_mask.astype(bool)
            if "tracked_query_count" in raw_payload:
                raw_payload["tracked_query_count"] = np.array(
                    [int(np.count_nonzero(selected_valid_mask))],
                    dtype=np.int32,
                )
            raw_payload["traj_query_visibility_reliable_mask"] = reliable_track_mask.astype(bool)
            raw_payload["traj_query_visibility_removed_mask"] = (~reliable_track_mask).astype(bool)
            raw_payload["traj_query_visibility_future_visible_ratio"] = np.asarray(
                gate_result["future_visible_ratio"], dtype=np.float32
            )
            raw_payload["traj_query_visibility_first_invalid_step"] = np.asarray(
                gate_result["first_invalid_step"], dtype=np.int32
            )
            raw_payload["traj_query_visibility_projected_in_bounds_mask"] = np.asarray(
                gate_result["projected_in_bounds_mask"], dtype=bool
            )

            np.savez(dst_path, **raw_payload)
            summaries.append(
                {
                    "sample_npz": str(sample_path),
                    "output_sample_npz": str(dst_path),
                    "query_frame": query_frame,
                    "segment_frame_indices": segment_frame_indices.astype(int).tolist(),
                    "gate_summary": gate_result["summary"],
                    "tracked_summary": gate_summary,
                    "track_mask_mode": str(args.track_mask_mode),
                    "original_valid_count": int(np.count_nonzero(original_valid_mask)),
                    "selected_valid_count": int(np.count_nonzero(selected_valid_mask)),
                }
            )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(summary_json),
                "query_frames": sorted(query_frames),
                "track_mask_mode": str(args.track_mask_mode),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
