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

from utils.external_wobble_diagnostics import compute_static_geometry_consistency
from utils.tracker_geometry_interaction_diagnostics import (
    compute_static_geometry_track_drift,
    summarize_tracker_geometry_interaction,
)


def _parse_query_frames(raw: str) -> list[int]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one query frame.")
    return [int(item) for item in items]


def _load_depth_stack(depth_dir: Path) -> np.ndarray:
    depth_paths = sorted(depth_dir.glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth .npy files found under {depth_dir}")
    return np.stack([np.load(path).astype(np.float32) for path in depth_paths], axis=0).astype(np.float32)


def _load_geom(geom_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    geom = np.load(geom_npz)
    try:
        intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)
    finally:
        geom.close()
    return intrinsics, extrinsics


def _load_tracker_sample(sample_npz: Path) -> dict[str, np.ndarray | int | None]:
    sample = np.load(sample_npz, allow_pickle=False)
    try:
        return {
            "keypoints": np.asarray(sample["keypoints"], dtype=np.float32),
            "traj_uvz": np.asarray(sample["traj_uvz"], dtype=np.float32),
            "traj_valid_mask": np.asarray(sample["traj_valid_mask"], dtype=bool),
            "valid_steps": np.asarray(sample["valid_steps"], dtype=bool) if "valid_steps" in sample else None,
            "segment_frame_indices": np.asarray(sample["segment_frame_indices"], dtype=np.int32).reshape(-1),
        }
    finally:
        sample.close()


def _build_tracker_summary(
    *,
    sample_npz: Path,
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    geom_stable_threshold_px: float,
    tracker_unstable_threshold_px: float,
    excess_threshold_px: float,
) -> dict[str, object]:
    sample = _load_tracker_sample(sample_npz)
    frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32)
    static_result = compute_static_geometry_track_drift(
        depth_frames[frame_indices],
        intrinsics[frame_indices],
        extrinsics[frame_indices],
        keypoints=np.asarray(sample["keypoints"], dtype=np.float32),
        query_frame=0,
        min_query_depth_m=0.01,
        min_border_dist_px=0.0,
    )
    summary = summarize_tracker_geometry_interaction(
        traj_uvz=np.asarray(sample["traj_uvz"], dtype=np.float32),
        keypoints=np.asarray(sample["keypoints"], dtype=np.float32),
        static_geometry_drift_px=np.asarray(static_result["final_query_reproj_drift_px"], dtype=np.float32),
        static_geometry_valid=np.asarray(static_result["final_query_reproj_valid"], dtype=bool),
        traj_valid_mask=np.asarray(sample["traj_valid_mask"], dtype=bool),
        valid_steps=None if sample["valid_steps"] is None else np.asarray(sample["valid_steps"], dtype=bool),
        geom_stable_threshold_px=float(geom_stable_threshold_px),
        tracker_unstable_threshold_px=float(tracker_unstable_threshold_px),
        excess_threshold_px=float(excess_threshold_px),
    )
    return {
        "tracker_final_drift_summary": summary["tracker_final_drift_summary"],
        "static_geometry_final_drift_summary": summary["static_geometry_final_drift_summary"],
        "excess_final_drift_summary": summary["excess_final_drift_summary"],
        "tracker_local_interaction_count": int(np.count_nonzero(summary["tracker_local_interaction_mask"])),
        "geometry_limited_count": int(np.count_nonzero(summary["geometry_limited_mask"])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs refined static-background geometry on q0/q4 smoke diagnostics."
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
        "--baseline_geom_npz",
        type=Path,
        required=True,
        help="Baseline geometry NPZ, typically ORB pose-only geom.",
    )
    parser.add_argument(
        "--refined_geom_npz",
        type=Path,
        required=True,
        help="Refined geometry NPZ produced by static-background pose refinement.",
    )
    parser.add_argument(
        "--trajectory_output_dir",
        type=Path,
        required=True,
        help="Existing trajectory output dir whose sample NPZ files will be reused for tracker smoke.",
    )
    parser.add_argument("--query_frames", type=str, default="0,4")
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--min_query_depth_m", type=float, default=0.2)
    parser.add_argument("--min_border_dist_px", type=float, default=60.0)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--geom_stable_threshold_px", type=float, default=1.0)
    parser.add_argument("--tracker_unstable_threshold_px", type=float, default=3.0)
    parser.add_argument("--excess_threshold_px", type=float, default=2.0)
    parser.add_argument("--max_global_disp_regression_px", type=float, default=0.25)
    parser.add_argument("--min_static_p95_gain_px", type=float, default=0.0)
    parser.add_argument("--min_geometry_limited_gain", type=int, default=0)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = args.case_dir
    camera_name = str(args.camera_name)
    depth_dir = args.depth_dir if args.depth_dir is not None else case_dir / "depth" / camera_name
    depth_frames = _load_depth_stack(depth_dir)
    base_intrinsics, base_extrinsics = _load_geom(args.baseline_geom_npz)
    refined_intrinsics, refined_extrinsics = _load_geom(args.refined_geom_npz)
    if base_intrinsics.shape != refined_intrinsics.shape:
        raise ValueError(f"Intrinsic shape mismatch: {base_intrinsics.shape} vs {refined_intrinsics.shape}")

    rows = []
    all_checks: list[bool] = []
    for query_frame in _parse_query_frames(args.query_frames):
        base_upstream = compute_static_geometry_consistency(
            depth_frames,
            base_intrinsics,
            base_extrinsics,
            query_frame=query_frame,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )
        refined_upstream = compute_static_geometry_consistency(
            depth_frames,
            refined_intrinsics,
            refined_extrinsics,
            query_frame=query_frame,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )
        sample_npz = args.trajectory_output_dir / "samples" / f"{camera_name}_{query_frame}.npz"
        if not sample_npz.is_file():
            raise FileNotFoundError(f"Missing sample NPZ: {sample_npz}")
        base_tracker = _build_tracker_summary(
            sample_npz=sample_npz,
            depth_frames=depth_frames,
            intrinsics=base_intrinsics,
            extrinsics=base_extrinsics,
            geom_stable_threshold_px=float(args.geom_stable_threshold_px),
            tracker_unstable_threshold_px=float(args.tracker_unstable_threshold_px),
            excess_threshold_px=float(args.excess_threshold_px),
        )
        refined_tracker = _build_tracker_summary(
            sample_npz=sample_npz,
            depth_frames=depth_frames,
            intrinsics=refined_intrinsics,
            extrinsics=refined_extrinsics,
            geom_stable_threshold_px=float(args.geom_stable_threshold_px),
            tracker_unstable_threshold_px=float(args.tracker_unstable_threshold_px),
            excess_threshold_px=float(args.excess_threshold_px),
        )

        global_disp_check = (
            float(refined_upstream["final_query_reproj_global_disp_px"])
            <= float(base_upstream["final_query_reproj_global_disp_px"]) + float(args.max_global_disp_regression_px)
        )
        static_p95_gain = (
            float(base_tracker["static_geometry_final_drift_summary"]["p95"])
            - float(refined_tracker["static_geometry_final_drift_summary"]["p95"])
        )
        geometry_limited_gain = int(base_tracker["geometry_limited_count"]) - int(
            refined_tracker["geometry_limited_count"]
        )
        static_p95_check = static_p95_gain >= float(args.min_static_p95_gain_px)
        geometry_limited_check = geometry_limited_gain >= int(args.min_geometry_limited_gain)
        pass_gate = bool(global_disp_check and static_p95_check and geometry_limited_check)
        all_checks.append(pass_gate)

        rows.append(
            {
                "query_frame": int(query_frame),
                "sample_npz": str(sample_npz),
                "baseline_upstream": {
                    "final_query_reproj_global_disp_px": float(base_upstream["final_query_reproj_global_disp_px"]),
                    "final_query_reproj_drift_p95_px": float(base_upstream["final_query_reproj_drift_p95_px"]),
                },
                "refined_upstream": {
                    "final_query_reproj_global_disp_px": float(refined_upstream["final_query_reproj_global_disp_px"]),
                    "final_query_reproj_drift_p95_px": float(refined_upstream["final_query_reproj_drift_p95_px"]),
                },
                "baseline_tracker_smoke": base_tracker,
                "refined_tracker_smoke": refined_tracker,
                "delta": {
                    "upstream_global_disp_px": float(
                        refined_upstream["final_query_reproj_global_disp_px"]
                        - base_upstream["final_query_reproj_global_disp_px"]
                    ),
                    "upstream_drift_p95_px": float(
                        refined_upstream["final_query_reproj_drift_p95_px"]
                        - base_upstream["final_query_reproj_drift_p95_px"]
                    ),
                    "static_geometry_p95_px": float(
                        refined_tracker["static_geometry_final_drift_summary"]["p95"]
                        - base_tracker["static_geometry_final_drift_summary"]["p95"]
                    ),
                    "geometry_limited_count": int(refined_tracker["geometry_limited_count"])
                    - int(base_tracker["geometry_limited_count"]),
                    "tracker_local_interaction_count": int(refined_tracker["tracker_local_interaction_count"])
                    - int(base_tracker["tracker_local_interaction_count"]),
                },
                "checks": {
                    "global_disp_check": bool(global_disp_check),
                    "static_p95_check": bool(static_p95_check),
                    "geometry_limited_check": bool(geometry_limited_check),
                    "pass_gate": pass_gate,
                },
            }
        )

    payload = {
        "case_dir": str(case_dir),
        "camera_name": camera_name,
        "depth_dir": str(depth_dir),
        "baseline_geom_npz": str(args.baseline_geom_npz),
        "refined_geom_npz": str(args.refined_geom_npz),
        "trajectory_output_dir": str(args.trajectory_output_dir),
        "query_frames": _parse_query_frames(args.query_frames),
        "gate_config": {
            "max_global_disp_regression_px": float(args.max_global_disp_regression_px),
            "min_static_p95_gain_px": float(args.min_static_p95_gain_px),
            "min_geometry_limited_gain": int(args.min_geometry_limited_gain),
        },
        "pass_gate": bool(all(all_checks)),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
