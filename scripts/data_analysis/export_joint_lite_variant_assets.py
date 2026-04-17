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
    smooth_extrinsics_w2c_moving_average,
    stabilize_depth_frames_temporal_median_reproject,
)


def _sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return int(digits[-1]), stem
    return 0, stem


def _collect_depth_paths(depth_dir: Path) -> list[Path]:
    depth_paths = [path for path in depth_dir.iterdir() if path.is_file() and path.suffix.lower() == ".npy"]
    depth_paths.sort(key=_sort_key)
    if not depth_paths:
        raise FileNotFoundError(f"No depth .npy files found under {depth_dir}")
    return depth_paths


def _load_depth_stack(depth_paths: list[Path]) -> np.ndarray:
    return np.stack([np.load(path).astype(np.float32) for path in depth_paths], axis=0).astype(np.float32)


def _finite_stat(values: np.ndarray, reducer: str) -> float | None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    if reducer == "median":
        return float(np.median(finite))
    if reducer == "p95":
        return float(np.percentile(finite, 95))
    raise ValueError(f"Unsupported reducer: {reducer}")


def _resolve_source_geom(case_dir: Path, camera_name: str, source_geom_npz: Path | None) -> Path:
    if source_geom_npz is not None:
        return source_geom_npz.resolve()
    orb_geom = case_dir / "geom" / f"geom_{camera_name}_orbslam3_rgbd_w2c.npz"
    if orb_geom.is_file():
        return orb_geom.resolve()
    return (case_dir / "geom" / f"geom_{camera_name}_official_w2c.npz").resolve()


def _load_geom(geom_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    if not geom_npz.is_file():
        raise FileNotFoundError(f"Missing geom npz: {geom_npz}")
    geom = np.load(geom_npz)
    try:
        intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)
    finally:
        geom.close()
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")
    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError(
            f"Frame count mismatch between intrinsics and extrinsics: {intrinsics.shape[0]} vs {extrinsics.shape[0]}"
        )
    return intrinsics, extrinsics


def _write_depth_stack(depth_dir: Path, *, depth_paths: list[Path], depth_stack: np.ndarray) -> None:
    depth_dir.mkdir(parents=True, exist_ok=True)
    if depth_stack.shape[0] != len(depth_paths):
        raise ValueError(f"Depth frame count mismatch: {depth_stack.shape[0]} vs {len(depth_paths)}")
    for src_path, depth_frame in zip(depth_paths, depth_stack):
        np.save(depth_dir / src_path.name, np.asarray(depth_frame, dtype=np.float32))


def _write_geom_npz(geom_npz: Path, *, intrinsics: np.ndarray, extrinsics: np.ndarray) -> None:
    geom_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        geom_npz,
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
    )


def _build_variant_dir(output_root: Path, variant_name: str, camera_name: str) -> tuple[Path, Path]:
    variant_dir = output_root / variant_name
    depth_dir = variant_dir / "depth" / camera_name
    geom_npz = variant_dir / "geom" / f"geom_{camera_name}_{variant_name}_w2c.npz"
    return depth_dir, geom_npz


def _write_variant_assets(
    *,
    output_root: Path,
    variant_name: str,
    camera_name: str,
    depth_paths: list[Path],
    depth_stack: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> dict[str, str]:
    depth_dir, geom_npz = _build_variant_dir(output_root, variant_name, camera_name)
    _write_depth_stack(depth_dir, depth_paths=depth_paths, depth_stack=depth_stack)
    _write_geom_npz(geom_npz, intrinsics=intrinsics, extrinsics=extrinsics)
    return {
        "variant_dir": str(depth_dir.parents[1]),
        "depth_dir": str(depth_dir),
        "geom_npz": str(geom_npz),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export joint-lite depth/geometry variant assets for controlled TraceForge reruns."
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
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Defaults to <case_dir>/_analysis_joint_lite/assets.",
    )
    parser.add_argument("--extr_smooth_radius", type=int, default=1)
    parser.add_argument("--dense_depth_radius", type=int, default=2)
    parser.add_argument("--dense_depth_min_support", type=int, default=3)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
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
        else (case_dir / "_analysis_joint_lite" / "assets").resolve()
    )

    depth_paths = _collect_depth_paths(source_depth_dir)
    depth_stack = _load_depth_stack(depth_paths)
    intrinsics, extrinsics = _load_geom(source_geom_npz)
    if depth_stack.shape[0] != intrinsics.shape[0]:
        raise ValueError(
            f"Frame count mismatch between depth and geometry: {depth_stack.shape[0]} vs {intrinsics.shape[0]}"
        )

    print(
        f"[joint-lite] loaded case={case_dir.name} camera={camera_name} "
        f"frames={depth_stack.shape[0]} depth_shape={tuple(depth_stack.shape[1:])}",
        flush=True,
    )
    print("[joint-lite] running dense depth stabilization", flush=True)
    dense_depth_result = stabilize_depth_frames_temporal_median_reproject(
        depth_stack,
        intrinsics,
        extrinsics,
        radius=int(args.dense_depth_radius),
        min_support=int(args.dense_depth_min_support),
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
    )
    print("[joint-lite] running extrinsics smoothing", flush=True)
    smoothed_extrinsics = smooth_extrinsics_w2c_moving_average(
        extrinsics,
        radius=int(args.extr_smooth_radius),
    )

    original_depth = np.asarray(depth_stack, dtype=np.float32)
    stabilized_depth = np.asarray(dense_depth_result["depth_frames"], dtype=np.float32)
    variants = {
        "orbslam3rgbd_densedepthtmr1": {
            "depth_stack": stabilized_depth,
            "extrinsics": extrinsics,
            "notes": ["ORB pose + dense depth temporal median reproject stabilization."],
        },
        "orbslam3rgbd_extrsm1": {
            "depth_stack": original_depth,
            "extrinsics": smoothed_extrinsics,
            "notes": [f"ORB pose + extrinsics moving-average smoothing radius={int(args.extr_smooth_radius)}."],
        },
        "orbslam3rgbd_jointlitev1": {
            "depth_stack": stabilized_depth,
            "extrinsics": smoothed_extrinsics,
            "notes": [
                "ORB pose + dense depth temporal median reproject stabilization + extrinsics moving-average smoothing."
            ],
        },
    }

    variant_payloads: dict[str, dict[str, object]] = {}
    for variant_name, variant in variants.items():
        print(f"[joint-lite] writing variant={variant_name}", flush=True)
        asset_paths = _write_variant_assets(
            output_root=output_root,
            variant_name=variant_name,
            camera_name=camera_name,
            depth_paths=depth_paths,
            depth_stack=np.asarray(variant["depth_stack"], dtype=np.float32),
            intrinsics=intrinsics,
            extrinsics=np.asarray(variant["extrinsics"], dtype=np.float32),
        )
        variant_payloads[variant_name] = {
            **asset_paths,
            "notes": list(variant["notes"]),
        }

    payload = {
        "case_dir": str(case_dir),
        "camera_name": camera_name,
        "source_depth_dir": str(source_depth_dir),
        "source_geom_npz": str(source_geom_npz),
        "output_root": str(output_root),
        "frame_count": int(depth_stack.shape[0]),
        "dense_depth_stabilization": {
            "radius": int(args.dense_depth_radius),
            "min_support": int(args.dense_depth_min_support),
            "replace_ratio_median": _finite_stat(dense_depth_result["replace_ratio"], "median"),
            "replace_ratio_p95": _finite_stat(dense_depth_result["replace_ratio"], "p95"),
            "replace_count_total": int(np.sum(np.asarray(dense_depth_result["replace_count"], dtype=np.int64))),
            "depth_delta_median_median_m": _finite_stat(dense_depth_result["depth_delta_median_m"], "median"),
            "depth_delta_p95_p95_m": _finite_stat(dense_depth_result["depth_delta_p95_m"], "p95"),
        },
        "extrinsics_smoothing": {
            "radius": int(args.extr_smooth_radius),
        },
        "variants": variant_payloads,
    }

    output_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    print(output_text, end="")
    summary_json = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else (output_root.parent / "joint_lite_assets_summary.json").resolve()
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
