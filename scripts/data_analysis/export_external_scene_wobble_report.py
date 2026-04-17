#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.scene_wobble_utils import compute_scene_wobble_summary


def _parse_query_frames(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        return None
    return {int(item) for item in items}


def _load_scene_meta(trajectory_output_dir: Path) -> dict[str, object]:
    meta_path = trajectory_output_dir / "scene_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text())


def _load_summary(
    sample_path: Path,
    *,
    args: argparse.Namespace,
    scene_meta: dict[str, object],
) -> dict[str, object]:
    data = np.load(sample_path)
    try:
        traj_uvz = np.asarray(data["traj_uvz"], dtype=np.float32)
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
        traj_valid_mask = (
            np.asarray(data["traj_valid_mask"], dtype=bool)
            if "traj_valid_mask" in data
            else np.ones(traj_uvz.shape[0], dtype=bool)
        )
        query_border_dist_px = (
            np.asarray(data["traj_query_border_dist_px"], dtype=np.float32)
            if "traj_query_border_dist_px" in data
            else None
        )
        valid_steps = np.asarray(data["valid_steps"], dtype=bool) if "valid_steps" in data else None
        query_frame = int(np.asarray(data["query_frame_index"]).reshape(-1)[0])
        dense_query_count = (
            int(np.asarray(data["dense_query_count"]).reshape(-1)[0])
            if "dense_query_count" in data
            else int(traj_uvz.shape[0])
        )
        tracked_query_count = (
            int(np.asarray(data["tracked_query_count"]).reshape(-1)[0])
            if "tracked_query_count" in data
            else int(traj_uvz.shape[0])
        )
    finally:
        data.close()

    if query_border_dist_px is not None and not np.any(query_border_dist_px > 0):
        query_border_dist_px = None

    image_height = scene_meta.get("height")
    image_width = scene_meta.get("width")

    summary = compute_scene_wobble_summary(
        traj_uvz,
        traj_valid_mask=traj_valid_mask,
        keypoints=keypoints,
        query_border_dist_px=query_border_dist_px,
        valid_steps=valid_steps,
        image_height=None if image_height is None else int(image_height),
        image_width=None if image_width is None else int(image_width),
        min_query_depth_m=float(args.min_query_depth_m),
        min_border_dist_px=float(args.min_border_dist_px),
        min_anchor_count=int(args.min_anchor_count),
        global_disp_threshold_px=float(args.global_disp_threshold_px),
    )
    return {
        "sample": sample_path.name,
        "query_frame": query_frame,
        "dense_query_count": dense_query_count,
        "tracked_query_count": tracked_query_count,
        "anchor_count": int(summary["anchor_count"]),
        "has_sufficient_anchors": bool(summary["has_sufficient_anchors"]),
        "geometry_unstable": bool(summary["geometry_unstable"]),
        "final_step_index": int(summary["final_step_index"]),
        "final_anchor_count": int(summary["final_anchor_count"]),
        "global_final_disp_px": float(summary["global_final_disp_px"]),
        "residual_final_p95_px": float(summary["residual_final_p95_px"]),
        "track_final_p95_px": float(summary["track_final_p95_px"]),
        "global_disp_p95_px": float(summary["global_disp_p95_px"]),
        "residual_disp_p95_px": float(summary["residual_disp_p95_px"]),
        "track_disp_p95_px": float(summary["track_disp_p95_px"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export fixed-view scene wobble diagnostics for external trajectory outputs."
    )
    parser.add_argument(
        "--trajectory_output_dir",
        type=Path,
        required=True,
        help="Path to one camera output dir, e.g. <episode>/trajectory_xxx/stereo_left.",
    )
    parser.add_argument(
        "--query_frames",
        type=str,
        default=None,
        help="Optional comma-separated query frame indices. Omit to scan all samples.",
    )
    parser.add_argument("--min_query_depth_m", type=float, default=0.2)
    parser.add_argument("--min_border_dist_px", type=float, default=60.0)
    parser.add_argument("--min_anchor_count", type=int, default=32)
    parser.add_argument("--global_disp_threshold_px", type=float, default=3.0)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sample_dir = args.trajectory_output_dir / "samples"
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing samples dir: {sample_dir}")

    requested_query_frames = _parse_query_frames(args.query_frames)
    sample_paths = sorted(sample_dir.glob("*.npz"))
    if requested_query_frames is not None:
        filtered_paths: list[Path] = []
        for path in sample_paths:
            stem_parts = path.stem.rsplit("_", 1)
            if len(stem_parts) != 2:
                continue
            try:
                query_frame = int(stem_parts[1])
            except ValueError:
                continue
            if query_frame in requested_query_frames:
                filtered_paths.append(path)
        sample_paths = filtered_paths

    if not sample_paths:
        raise FileNotFoundError(f"No matching sample NPZ files found under {sample_dir}")

    scene_meta = _load_scene_meta(args.trajectory_output_dir)
    rows = [_load_summary(path, args=args, scene_meta=scene_meta) for path in sample_paths]
    rows.sort(key=lambda item: int(item["query_frame"]))

    payload = {
        "trajectory_output_dir": str(args.trajectory_output_dir),
        "min_query_depth_m": float(args.min_query_depth_m),
        "min_border_dist_px": float(args.min_border_dist_px),
        "min_anchor_count": int(args.min_anchor_count),
        "global_disp_threshold_px": float(args.global_disp_threshold_px),
        "rows": rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
