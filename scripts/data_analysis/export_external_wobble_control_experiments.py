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
    build_query_anchor_bundle,
    compute_static_geometry_consistency,
    estimate_temporal_median_world_points,
    freeze_extrinsics_w2c_to_query_frame,
    smooth_extrinsics_w2c_moving_average,
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
        description="Run geometry-only control experiments to separate depth-dominant vs extrinsics-dominant wobble."
    )
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument("--camera_name", type=str, default="stereo_left")
    parser.add_argument("--query_frames", type=str, default="0,4")
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--min_query_depth_m", type=float, default=0.2)
    parser.add_argument("--min_border_dist_px", type=float, default=60.0)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--extr_smooth_radius", type=int, default=1)
    parser.add_argument("--depth_median_reproj_tol_px", type=float, default=3.0)
    parser.add_argument("--depth_median_min_support", type=int, default=3)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser


def _summarize_variant(
    *,
    variant_name: str,
    metrics: dict[str, object],
    replace_mask: np.ndarray | None = None,
    support_counts: np.ndarray | None = None,
) -> dict[str, object]:
    payload = {
        "variant": variant_name,
        "anchor_count": int(metrics["anchor_count"]),
        "final_query_reproj_global_disp_px": float(metrics["final_query_reproj_global_disp_px"]),
        "final_query_reproj_drift_median_px": float(metrics["final_query_reproj_drift_median_px"]),
        "final_query_reproj_drift_p95_px": float(metrics["final_query_reproj_drift_p95_px"]),
        "query_reproj_global_disp_summary": metrics["query_reproj_global_disp_summary"],
        "query_reproj_drift_median_summary": metrics["query_reproj_drift_median_summary"],
        "query_reproj_drift_p95_summary": metrics["query_reproj_drift_p95_summary"],
    }
    if replace_mask is not None:
        payload["depth_variant_replace_frac"] = float(np.count_nonzero(replace_mask) / max(len(replace_mask), 1))
    if support_counts is not None:
        payload["depth_variant_support_summary"] = {
            "median": float(np.median(support_counts)),
            "p95": float(np.percentile(support_counts, 95)),
            "max": int(np.max(support_counts)),
        }
    return payload


def main() -> None:
    args = build_parser().parse_args()
    case_dir = args.case_dir
    camera_name = str(args.camera_name)
    depth_dir = case_dir / "depth" / camera_name
    geom_npz = case_dir / "geom" / f"geom_{camera_name}_official_w2c.npz"
    if not geom_npz.is_file():
        raise FileNotFoundError(f"Missing geom npz: {geom_npz}")

    depth_frames = load_depth_stack(depth_dir)
    geom = np.load(geom_npz)
    try:
        intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)
    finally:
        geom.close()

    rows = []
    for query_frame in parse_query_frames(args.query_frames):
        baseline_bundle = build_query_anchor_bundle(
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
        baseline_metrics = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=query_frame,
            query_anchor_bundle=baseline_bundle,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )

        extr_smooth = smooth_extrinsics_w2c_moving_average(extrinsics, radius=int(args.extr_smooth_radius))
        smooth_bundle = build_query_anchor_bundle(
            depth_frames,
            intrinsics,
            extr_smooth,
            query_frame=query_frame,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )
        smooth_metrics = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            extr_smooth,
            query_frame=query_frame,
            query_anchor_bundle=smooth_bundle,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )

        extr_frozen = freeze_extrinsics_w2c_to_query_frame(extrinsics, query_frame=query_frame)
        frozen_bundle = build_query_anchor_bundle(
            depth_frames,
            intrinsics,
            extr_frozen,
            query_frame=query_frame,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )
        frozen_metrics = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            extr_frozen,
            query_frame=query_frame,
            query_anchor_bundle=frozen_bundle,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )

        depth_temporal = estimate_temporal_median_world_points(
            depth_frames,
            intrinsics,
            extrinsics,
            query_anchor_bundle=baseline_bundle,
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
            reproj_tol_px=float(args.depth_median_reproj_tol_px),
            min_support=int(args.depth_median_min_support),
        )
        depth_bundle = dict(baseline_bundle)
        depth_bundle["world_points"] = np.asarray(depth_temporal["world_points"], dtype=np.float32)
        depth_metrics = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=query_frame,
            query_anchor_bundle=depth_bundle,
            grid_size=int(args.grid_size),
            min_query_depth_m=float(args.min_query_depth_m),
            min_border_dist_px=float(args.min_border_dist_px),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
        )

        variants = [
            _summarize_variant(variant_name="baseline", metrics=baseline_metrics),
            _summarize_variant(variant_name=f"extrinsics_smooth_r{int(args.extr_smooth_radius)}", metrics=smooth_metrics),
            _summarize_variant(variant_name="extrinsics_freeze_query", metrics=frozen_metrics),
            _summarize_variant(
                variant_name="depth_temporal_median_world_v1",
                metrics=depth_metrics,
                replace_mask=np.asarray(depth_temporal["replace_mask"], dtype=bool),
                support_counts=np.asarray(depth_temporal["support_counts"], dtype=np.int32),
            ),
        ]
        baseline_final_median = float(baseline_metrics["final_query_reproj_drift_median_px"])
        baseline_final_p95 = float(baseline_metrics["final_query_reproj_drift_p95_px"])
        for item in variants:
            item["delta_final_query_reproj_drift_median_px"] = (
                float(item["final_query_reproj_drift_median_px"]) - baseline_final_median
            )
            item["delta_final_query_reproj_drift_p95_px"] = (
                float(item["final_query_reproj_drift_p95_px"]) - baseline_final_p95
            )
        rows.append(
            {
                "query_frame": int(query_frame),
                "variants": variants,
            }
        )

    payload = {
        "case_dir": str(case_dir),
        "camera_name": camera_name,
        "query_frames": parse_query_frames(args.query_frames),
        "grid_size": int(args.grid_size),
        "min_query_depth_m": float(args.min_query_depth_m),
        "min_border_dist_px": float(args.min_border_dist_px),
        "extr_smooth_radius": int(args.extr_smooth_radius),
        "depth_median_reproj_tol_px": float(args.depth_median_reproj_tol_px),
        "depth_median_min_support": int(args.depth_median_min_support),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
