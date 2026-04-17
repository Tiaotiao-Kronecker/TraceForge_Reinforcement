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

from utils.static_geometry_refinement import audit_static_geometry_heavy_tail


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit static-geometry heavy tails by frame and spatial cell for one case."
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
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--cell_size_px", type=int, default=64)
    parser.add_argument("--tail_threshold_px", type=float, default=20.0)
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--top_k_cells", type=int, default=8)
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

    depth_frames = _load_depth_stack(depth_dir)
    geom = np.load(geom_npz)
    try:
        intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)
    finally:
        geom.close()

    query_reports = []
    for query_frame in _parse_query_frames(args.query_frames):
        query_reports.append(
            audit_static_geometry_heavy_tail(
                depth_frames,
                intrinsics,
                extrinsics,
                query_frame=query_frame,
                grid_size=int(args.grid_size),
                min_query_depth_m=float(args.min_query_depth_m),
                min_border_dist_px=float(args.min_border_dist_px),
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                cell_size_px=int(args.cell_size_px),
                tail_threshold_px=float(args.tail_threshold_px),
                top_k_frames=int(args.top_k_frames),
                top_k_cells=int(args.top_k_cells),
            )
        )

    payload = {
        "case_dir": str(case_dir),
        "camera_name": camera_name,
        "depth_dir": str(depth_dir),
        "geom_npz": str(geom_npz),
        "depth_shape": list(depth_frames.shape),
        "query_frames": _parse_query_frames(args.query_frames),
        "grid_size": int(args.grid_size),
        "min_query_depth_m": float(args.min_query_depth_m),
        "min_border_dist_px": float(args.min_border_dist_px),
        "cell_size_px": int(args.cell_size_px),
        "tail_threshold_px": float(args.tail_threshold_px),
        "query_reports": query_reports,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
