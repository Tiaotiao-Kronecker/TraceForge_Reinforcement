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
from utils.static_geometry_refinement import refine_extrinsics_w2c_static_background


def _parse_query_frames(raw: str) -> list[int]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one query frame.")
    return [int(item) for item in items]


def _sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return int(digits[-1]), stem
    return 0, stem


def _load_depth_stack(depth_dir: Path) -> np.ndarray:
    depth_paths = [path for path in depth_dir.iterdir() if path.is_file() and path.suffix.lower() == ".npy"]
    depth_paths.sort(key=_sort_key)
    if not depth_paths:
        raise FileNotFoundError(f"No depth .npy files found under {depth_dir}")
    return np.stack([np.load(path).astype(np.float32) for path in depth_paths], axis=0).astype(np.float32)


def _resolve_source_geom(case_dir: Path, camera_name: str, source_geom_npz: Path | None) -> Path:
    if source_geom_npz is not None:
        return source_geom_npz.resolve()
    orb_geom = case_dir / "geom" / f"geom_{camera_name}_orbslam3_rgbd_w2c.npz"
    if orb_geom.is_file():
        return orb_geom.resolve()
    return (case_dir / "geom" / f"geom_{camera_name}_official_w2c.npz").resolve()


def _write_geom_npz(geom_npz: Path, *, intrinsics: np.ndarray, extrinsics: np.ndarray) -> None:
    geom_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        geom_npz,
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export static-background pose refinement v1 geom assets and summary for one case."
    )
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument("--camera_name", type=str, default="stereo_left")
    parser.add_argument(
        "--source_depth_dir",
        type=Path,
        default=None,
        help="Defaults to <case_dir>/depth/<camera_name>.",
    )
    parser.add_argument(
        "--source_geom_npz",
        type=Path,
        default=None,
        help="Defaults to ORB geom when present, otherwise official geom.",
    )
    parser.add_argument("--query_frames", type=str, default="0,4")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Defaults to <case_dir>/_analysis_static_bg_refine.",
    )
    parser.add_argument(
        "--refined_geom_npz",
        type=Path,
        default=None,
        help="Optional explicit refined geom path.",
    )
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--min_query_depth_m", type=float, default=0.2)
    parser.add_argument("--min_border_dist_px", type=float, default=60.0)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_target_border_dist_px", type=float, default=12.0)
    parser.add_argument("--max_depth_error_m", type=float, default=0.20)
    parser.add_argument("--max_world_error_m", type=float, default=0.20)
    parser.add_argument("--max_query_reproj_error_px", type=float, default=6.0)
    parser.add_argument("--min_correspondences", type=int, default=256)
    parser.add_argument("--temporal_smooth_radius", type=int, default=1)
    parser.add_argument("--temporal_regularization_weight", type=float, default=0.25)
    parser.add_argument("--max_translation_delta_m", type=float, default=0.05)
    parser.add_argument("--max_rotation_delta_deg", type=float, default=2.0)
    parser.add_argument("--summary_json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = args.case_dir.resolve()
    camera_name = str(args.camera_name)
    source_depth_dir = (
        args.source_depth_dir.resolve()
        if args.source_depth_dir is not None
        else (case_dir / "depth" / camera_name).resolve()
    )
    source_geom_npz = _resolve_source_geom(case_dir, camera_name, args.source_geom_npz)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (case_dir / "_analysis_static_bg_refine").resolve()
    )
    refined_geom_npz = (
        args.refined_geom_npz.resolve()
        if args.refined_geom_npz is not None
        else (
            output_root
            / "assets"
            / "orbslam3rgbd_bgrefinev1"
            / "geom"
            / f"geom_{camera_name}_orbslam3rgbd_bgrefinev1_w2c.npz"
        ).resolve()
    )
    summary_json = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else (output_root / "static_bg_pose_refinement_summary.json").resolve()
    )

    depth_frames = _load_depth_stack(source_depth_dir)
    geom = np.load(source_geom_npz)
    try:
        intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)
    finally:
        geom.close()
    if depth_frames.shape[0] != intrinsics.shape[0]:
        raise ValueError(
            f"Frame count mismatch between depth and geometry: {depth_frames.shape[0]} vs {intrinsics.shape[0]}"
        )

    query_frames = _parse_query_frames(args.query_frames)
    result = refine_extrinsics_w2c_static_background(
        depth_frames,
        intrinsics,
        extrinsics,
        query_frames=query_frames,
        grid_size=int(args.grid_size),
        min_query_depth_m=float(args.min_query_depth_m),
        min_border_dist_px=float(args.min_border_dist_px),
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
        min_target_border_dist_px=float(args.min_target_border_dist_px),
        max_depth_error_m=float(args.max_depth_error_m),
        max_world_error_m=float(args.max_world_error_m),
        max_query_reproj_error_px=float(args.max_query_reproj_error_px),
        min_correspondences=int(args.min_correspondences),
        temporal_smooth_radius=int(args.temporal_smooth_radius),
        temporal_regularization_weight=float(args.temporal_regularization_weight),
        max_translation_delta_m=float(args.max_translation_delta_m),
        max_rotation_delta_deg=float(args.max_rotation_delta_deg),
    )
    refined_extrinsics = np.asarray(result["extrinsics_w2c"], dtype=np.float32)
    _write_geom_npz(refined_geom_npz, intrinsics=intrinsics, extrinsics=refined_extrinsics)

    baseline_temporal = compute_extrinsics_temporal_metrics(extrinsics)
    refined_temporal = compute_extrinsics_temporal_metrics(refined_extrinsics)
    query_reports = []
    for query_frame in query_frames:
        baseline_metrics = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=query_frame,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )
        refined_metrics = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            refined_extrinsics,
            query_frame=query_frame,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )
        query_reports.append(
            {
                "query_frame": int(query_frame),
                "baseline_final_query_reproj_global_disp_px": float(
                    baseline_metrics["final_query_reproj_global_disp_px"]
                ),
                "refined_final_query_reproj_global_disp_px": float(
                    refined_metrics["final_query_reproj_global_disp_px"]
                ),
                "delta_final_query_reproj_global_disp_px": float(
                    refined_metrics["final_query_reproj_global_disp_px"]
                    - baseline_metrics["final_query_reproj_global_disp_px"]
                ),
                "baseline_final_query_reproj_drift_p95_px": float(
                    baseline_metrics["final_query_reproj_drift_p95_px"]
                ),
                "refined_final_query_reproj_drift_p95_px": float(
                    refined_metrics["final_query_reproj_drift_p95_px"]
                ),
                "delta_final_query_reproj_drift_p95_px": float(
                    refined_metrics["final_query_reproj_drift_p95_px"]
                    - baseline_metrics["final_query_reproj_drift_p95_px"]
                ),
                "baseline_query_reproj_drift_p95_summary": baseline_metrics["query_reproj_drift_p95_summary"],
                "refined_query_reproj_drift_p95_summary": refined_metrics["query_reproj_drift_p95_summary"],
            }
        )

    payload = {
        "case_dir": str(case_dir),
        "camera_name": camera_name,
        "source_depth_dir": str(source_depth_dir),
        "source_geom_npz": str(source_geom_npz),
        "refined_geom_npz": str(refined_geom_npz),
        "query_frames": query_frames,
        "frame_count": int(depth_frames.shape[0]),
        "grid_size": int(args.grid_size),
        "min_query_depth_m": float(args.min_query_depth_m),
        "min_border_dist_px": float(args.min_border_dist_px),
        "min_target_border_dist_px": float(args.min_target_border_dist_px),
        "max_depth_error_m": float(args.max_depth_error_m),
        "max_world_error_m": float(args.max_world_error_m),
        "max_query_reproj_error_px": float(args.max_query_reproj_error_px),
        "min_correspondences": int(args.min_correspondences),
        "temporal_smooth_radius": int(args.temporal_smooth_radius),
        "temporal_regularization_weight": float(args.temporal_regularization_weight),
        "max_translation_delta_m": float(args.max_translation_delta_m),
        "max_rotation_delta_deg": float(args.max_rotation_delta_deg),
        "baseline_temporal_metrics": {
            "camera_center_path_length_m": float(baseline_temporal["camera_center_path_length_m"]),
            "step_translation_summary": baseline_temporal["step_translation_summary"],
            "step_rotation_summary": baseline_temporal["step_rotation_summary"],
            "jerk_translation_summary": baseline_temporal["jerk_translation_summary"],
            "jerk_rotation_summary": baseline_temporal["jerk_rotation_summary"],
        },
        "refined_temporal_metrics": {
            "camera_center_path_length_m": float(refined_temporal["camera_center_path_length_m"]),
            "step_translation_summary": refined_temporal["step_translation_summary"],
            "step_rotation_summary": refined_temporal["step_rotation_summary"],
            "jerk_translation_summary": refined_temporal["jerk_translation_summary"],
            "jerk_rotation_summary": refined_temporal["jerk_rotation_summary"],
        },
        "refinement": {
            "support_count_summary": result["support_count_summary"],
            "fit_median_summary": result["fit_median_summary"],
            "fit_p95_summary": result["fit_p95_summary"],
            "translation_delta_summary": result["translation_delta_summary"],
            "rotation_delta_summary": result["rotation_delta_summary"],
            "frame_reports": result["frame_reports"],
        },
        "query_reports": query_reports,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
