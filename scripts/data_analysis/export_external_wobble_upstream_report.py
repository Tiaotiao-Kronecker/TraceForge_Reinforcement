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

from utils.external_wobble_diagnostics import (
    compute_extrinsics_temporal_metrics,
    compute_static_geometry_consistency,
)


def parse_query_frames(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one query frame index.")
    return [int(item) for item in values]


def load_depth_stack(depth_dir: Path) -> np.ndarray:
    depth_paths = sorted(depth_dir.glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth .npy files found under {depth_dir}")
    return np.stack([np.load(path).astype(np.float32) for path in depth_paths], axis=0).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export upstream external depth/extrinsics wobble diagnostics for one motion-window case."
    )
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument("--camera_name", type=str, default="stereo_left")
    parser.add_argument(
        "--depth_dir",
        type=Path,
        default=None,
        help="Optional depth directory override. Defaults to <case_dir>/depth/<camera_name>.",
    )
    parser.add_argument(
        "--geom_npz",
        type=Path,
        default=None,
        help="Optional geometry NPZ override. Defaults to <case_dir>/geom/geom_<camera_name>_official_w2c.npz.",
    )
    parser.add_argument("--query_frames", type=str, default="0,4")
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--min_query_depth_m", type=float, default=0.2)
    parser.add_argument("--min_border_dist_px", type=float, default=60.0)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = args.case_dir
    camera_name = str(args.camera_name)
    depth_dir = args.depth_dir if args.depth_dir is not None else case_dir / "depth" / camera_name
    geom_npz = (
        args.geom_npz
        if args.geom_npz is not None
        else case_dir / "geom" / f"geom_{camera_name}_official_w2c.npz"
    )
    if not geom_npz.is_file():
        raise FileNotFoundError(f"Missing geom npz: {geom_npz}")

    depth_frames = load_depth_stack(depth_dir)
    geom = np.load(geom_npz)
    try:
        intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)
    finally:
        geom.close()

    extr_metrics = compute_extrinsics_temporal_metrics(extrinsics)
    query_reports = []
    for query_frame in parse_query_frames(args.query_frames):
        query_reports.append(
            compute_static_geometry_consistency(
                depth_frames,
                intrinsics,
                extrinsics,
                query_frame=query_frame,
                grid_size=int(args.grid_size),
                min_query_depth_m=float(args.min_query_depth_m),
                min_border_dist_px=float(args.min_border_dist_px),
            )
        )

    payload = {
        "case_dir": str(case_dir),
        "camera_name": camera_name,
        "depth_dir": str(depth_dir),
        "geom_npz": str(geom_npz),
        "depth_shape": list(depth_frames.shape),
        "extrinsics": {
            "frame_count": int(extr_metrics["frame_count"]),
            "camera_center_path_length_m": float(extr_metrics["camera_center_path_length_m"]),
            "step_translation_summary": extr_metrics["step_translation_summary"],
            "step_rotation_summary": extr_metrics["step_rotation_summary"],
            "jerk_translation_summary": extr_metrics["jerk_translation_summary"],
            "jerk_rotation_summary": extr_metrics["jerk_rotation_summary"],
        },
        "query_reports": [
            {
                "query_frame": int(item["query_frame"]),
                "grid_size": int(item["grid_size"]),
                "anchor_count": int(item["anchor_count"]),
                "final_depth_error_median_m": float(item["final_depth_error_median_m"]),
                "final_depth_error_p95_m": float(item["final_depth_error_p95_m"]),
                "final_world_error_median_m": float(item["final_world_error_median_m"]),
                "final_world_error_p95_m": float(item["final_world_error_p95_m"]),
                "final_query_reproj_global_disp_px": float(item["final_query_reproj_global_disp_px"]),
                "final_query_reproj_drift_median_px": float(item["final_query_reproj_drift_median_px"]),
                "final_query_reproj_drift_p95_px": float(item["final_query_reproj_drift_p95_px"]),
                "depth_error_median_summary": item["depth_error_median_summary"],
                "depth_error_p95_summary": item["depth_error_p95_summary"],
                "world_error_median_summary": item["world_error_median_summary"],
                "world_error_p95_summary": item["world_error_p95_summary"],
                "query_reproj_global_disp_summary": item["query_reproj_global_disp_summary"],
                "query_reproj_drift_median_summary": item["query_reproj_drift_median_summary"],
                "query_reproj_drift_p95_summary": item["query_reproj_drift_p95_summary"],
            }
            for item in query_reports
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
