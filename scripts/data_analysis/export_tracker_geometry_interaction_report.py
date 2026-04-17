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

from utils.tracker_geometry_interaction_diagnostics import (
    compute_static_geometry_track_drift,
    summarize_tracker_geometry_interaction,
)


def _load_depth_segment(depth_dir: Path, frame_indices: np.ndarray) -> np.ndarray:
    depth_paths = sorted(depth_dir.glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth .npy files found under {depth_dir}")
    max_index = int(np.max(frame_indices)) if frame_indices.size > 0 else -1
    if max_index >= len(depth_paths):
        raise IndexError(f"segment_frame_indices max={max_index} exceeds depth frame count {len(depth_paths)}")
    return np.stack(
        [np.load(depth_paths[int(frame_idx)]).astype(np.float32) for frame_idx in frame_indices],
        axis=0,
    ).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare tracker fixed-view drift against geometry-only drift to isolate tracker/local interaction."
    )
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument("--sample_npz", type=Path, required=True)
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
    parser.add_argument("--geom_stable_threshold_px", type=float, default=1.0)
    parser.add_argument("--tracker_unstable_threshold_px", type=float, default=3.0)
    parser.add_argument("--excess_threshold_px", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = args.case_dir
    sample_npz = args.sample_npz
    camera_name = str(args.camera_name)
    depth_dir = args.depth_dir if args.depth_dir is not None else case_dir / "depth" / camera_name
    geom_npz = (
        args.geom_npz
        if args.geom_npz is not None
        else case_dir / "geom" / f"geom_{camera_name}_official_w2c.npz"
    )
    if not geom_npz.is_file():
        raise FileNotFoundError(f"Missing geom npz: {geom_npz}")
    if not sample_npz.is_file():
        raise FileNotFoundError(f"Missing sample npz: {sample_npz}")

    sample = np.load(sample_npz, allow_pickle=False)
    try:
        keypoints = np.asarray(sample["keypoints"], dtype=np.float32)
        traj_uvz = np.asarray(sample["traj_uvz"], dtype=np.float32)
        traj_valid_mask = np.asarray(sample["traj_valid_mask"], dtype=bool)
        valid_steps = np.asarray(sample["valid_steps"], dtype=bool) if "valid_steps" in sample else None
        segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32).reshape(-1)
        query_depth_edge_risk_mask = np.asarray(
            sample["traj_query_depth_edge_risk_mask"], dtype=bool
        ) if "traj_query_depth_edge_risk_mask" in sample else np.zeros(keypoints.shape[0], dtype=bool)
        query_border_dist_px = np.asarray(
            sample["traj_query_border_dist_px"], dtype=np.float32
        ) if "traj_query_border_dist_px" in sample else np.full(keypoints.shape[0], np.nan, dtype=np.float32)
        query_depth_temporal_replace_mask = np.asarray(
            sample["traj_query_depth_temporal_replace_mask"], dtype=bool
        ) if "traj_query_depth_temporal_replace_mask" in sample else np.zeros(keypoints.shape[0], dtype=bool)
    finally:
        sample.close()

    depth_segment = _load_depth_segment(depth_dir, segment_frame_indices)
    geom = np.load(geom_npz)
    try:
        intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)[segment_frame_indices]
        extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)[segment_frame_indices]
    finally:
        geom.close()

    static_result = compute_static_geometry_track_drift(
        depth_segment,
        intrinsics,
        extrinsics,
        keypoints=keypoints,
        query_frame=0,
        min_query_depth_m=0.01,
        min_border_dist_px=0.0,
    )
    summary = summarize_tracker_geometry_interaction(
        traj_uvz=traj_uvz,
        keypoints=keypoints,
        static_geometry_drift_px=np.asarray(static_result["final_query_reproj_drift_px"], dtype=np.float32),
        static_geometry_valid=np.asarray(static_result["final_query_reproj_valid"], dtype=bool),
        traj_valid_mask=traj_valid_mask,
        valid_steps=valid_steps,
        geom_stable_threshold_px=float(args.geom_stable_threshold_px),
        tracker_unstable_threshold_px=float(args.tracker_unstable_threshold_px),
        excess_threshold_px=float(args.excess_threshold_px),
    )

    excess = np.asarray(summary["excess_final_drift_px"], dtype=np.float32)
    compare_mask = np.asarray(summary["tracker_final_valid"], dtype=bool) & np.asarray(
        summary["static_geometry_final_valid"], dtype=bool
    )
    candidate_indices = np.flatnonzero(compare_mask & np.isfinite(excess))
    order = candidate_indices[np.argsort(-excess[candidate_indices], kind="stable")]
    top_k = max(int(args.top_k), 0)
    top_candidates = []
    for track_idx in order[:top_k]:
        top_candidates.append(
            {
                "track_index": int(track_idx),
                "keypoint_xy": keypoints[track_idx].astype(float).tolist(),
                "tracker_final_drift_px": float(summary["tracker_final_drift_px"][track_idx]),
                "static_geometry_final_drift_px": float(summary["static_geometry_final_drift_px"][track_idx]),
                "excess_final_drift_px": float(summary["excess_final_drift_px"][track_idx]),
                "tracker_local_interaction": bool(summary["tracker_local_interaction_mask"][track_idx]),
                "geometry_limited": bool(summary["geometry_limited_mask"][track_idx]),
                "query_depth_edge_risk": bool(query_depth_edge_risk_mask[track_idx]),
                "query_border_dist_px": float(query_border_dist_px[track_idx]),
                "query_depth_temporal_replaced": bool(query_depth_temporal_replace_mask[track_idx]),
            }
        )

    payload = {
        "case_dir": str(case_dir),
        "sample_npz": str(sample_npz),
        "camera_name": camera_name,
        "depth_dir": str(depth_dir),
        "geom_npz": str(geom_npz),
        "query_frame_index": int(segment_frame_indices[0]) if segment_frame_indices.size > 0 else None,
        "segment_frame_count": int(segment_frame_indices.shape[0]),
        "final_step_index": int(summary["final_step_index"]),
        "geom_stable_threshold_px": float(args.geom_stable_threshold_px),
        "tracker_unstable_threshold_px": float(args.tracker_unstable_threshold_px),
        "excess_threshold_px": float(args.excess_threshold_px),
        "track_count": int(keypoints.shape[0]),
        "tracker_final_drift_summary": summary["tracker_final_drift_summary"],
        "static_geometry_final_drift_summary": summary["static_geometry_final_drift_summary"],
        "excess_final_drift_summary": summary["excess_final_drift_summary"],
        "tracker_local_interaction_count": int(np.count_nonzero(summary["tracker_local_interaction_mask"])),
        "geometry_limited_count": int(np.count_nonzero(summary["geometry_limited_mask"])),
        "top_excess_tracks": top_candidates,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
