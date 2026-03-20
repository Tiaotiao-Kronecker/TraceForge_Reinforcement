#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CURRENT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_ROOT))

from scripts.data_analysis.benchmark_depth_volatility_optimization import (  # noqa: E402
    ensure_query_frame_schedule,
    parse_camera_names,
    resolve_traj_filter_profile,
)
from utils.traceforge_artifact_utils import (  # noqa: E402
    SceneReader,
    normalize_sample_data,
    traj_uvz_to_world,
)


BASELINE_SUPPORT_GRID_RATIO = 0.8
DEFAULT_SUPPORT_GRID_RATIOS = (0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0)
QUERY_PREFILTER_MODE_OFF = "off"
RESULT_JSON_BASENAME = "benchmark_results.json"
SUMMARY_MD_BASENAME = "benchmark_summary.md"


def parse_args() -> argparse.Namespace:
    default_output_root = (
        CURRENT_REPO_ROOT
        / "data_tmp"
        / "inference_variant_benchmarks"
        / time.strftime("%Y%m%d_%H%M%S")
    )

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark support-grid-only inference variants inside the current repo "
            "and compare their end-to-end runtime and saved trajectory outputs."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=None,
        help="Episode directory, e.g. /data1/yaoxuran/press_one_button_demo_v1/episode_00000",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        default="varied_camera_1,varied_camera_3",
        help="Comma-separated camera names. Defaults to one external and one wrist-like camera.",
    )
    parser.add_argument(
        "--support-grid-ratios",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SUPPORT_GRID_RATIOS),
        help=(
            "Comma-separated support_grid_ratio sweep. Must include 0.8 as the baseline, "
            "for example 0.8,0.6,0.4,0.2,0.1,0.05,0.0."
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset of generated variant names. "
            "Generated names are baseline and support_rXXX style names."
        ),
    )
    parser.add_argument(
        "--traj-filter-profile",
        type=str,
        default="auto",
        help=(
            "Trajectory filter profile. 'auto' maps wrist-like camera names to "
            "wrist_manipulator_top95 and others to external."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CURRENT_REPO_ROOT / "checkpoints" / "tapip3d_final.pth",
        help="Checkpoint used for all variants.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device passed to infer.py, e.g. cuda:1",
    )
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1, help="Load stride passed to infer.py")
    parser.add_argument("--max-num-frames", type=int, default=512)
    parser.add_argument("--future-len", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument(
        "--filter-level",
        type=str,
        default="standard",
        choices=["none", "basic", "standard", "strict"],
    )
    parser.add_argument(
        "--keyframes-per-sec-min",
        type=int,
        default=2,
        help="Shared schedule minimum query frames per second.",
    )
    parser.add_argument(
        "--keyframes-per-sec-max",
        type=int,
        default=3,
        help="Shared schedule maximum query frames per second.",
    )
    parser.add_argument(
        "--keyframe-seed",
        type=int,
        default=0,
        help="Base seed for deterministic shared schedule generation.",
    )
    parser.add_argument(
        "--fallback-episode-fps",
        type=float,
        default=0.0,
        help="Used only if trajectory_valid.h5 does not carry a root attr 'fps'.",
    )
    parser.add_argument(
        "--external-geom-name",
        type=str,
        default="trajectory_valid.h5",
        help="Per-episode geometry filename.",
    )
    parser.add_argument(
        "--external-extr-mode",
        type=str,
        default="w2c",
        choices=["w2c", "c2w"],
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Measured runs per variant/camera after warmup.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per variant/camera before measurements.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Directory used for manifests, per-case json, summaries, and kept artifacts.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep all generated TraceForge outputs under output-root/artifacts.",
    )
    parser.add_argument(
        "--run-visual-verification",
        action="store_true",
        help="Generate verification PNGs on the first common query frame for each camera.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--query-frame-schedule-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def parse_variant_names(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("variants must contain at least one name")
    return values


def _normalize_support_grid_ratio(value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError("support_grid_ratio must be >= 0")
    if math.isclose(value, BASELINE_SUPPORT_GRID_RATIO, rel_tol=0.0, abs_tol=1e-6):
        return float(BASELINE_SUPPORT_GRID_RATIO)
    return float(round(value, 6))


def parse_support_grid_ratios(raw: str) -> list[float]:
    ratios: list[float] = []
    seen: set[float] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        ratio = _normalize_support_grid_ratio(float(item))
        if ratio in seen:
            continue
        seen.add(ratio)
        ratios.append(ratio)
    if not ratios:
        raise ValueError("support-grid-ratios must contain at least one value")
    if BASELINE_SUPPORT_GRID_RATIO not in seen:
        raise ValueError("support-grid-ratios must include 0.8 as the baseline")
    return ratios


def format_support_variant_name(support_grid_ratio: float) -> str:
    ratio = _normalize_support_grid_ratio(support_grid_ratio)
    if math.isclose(ratio, BASELINE_SUPPORT_GRID_RATIO, rel_tol=0.0, abs_tol=1e-6):
        return "baseline"

    percentage = ratio * 100.0
    rounded_percentage = int(round(percentage))
    if math.isclose(percentage, float(rounded_percentage), rel_tol=0.0, abs_tol=1e-6):
        return f"support_r{rounded_percentage:03d}"

    permille = int(round(ratio * 1000.0))
    return f"support_r{permille:04d}"


def resolve_requested_support_grid_size(grid_size: int, support_grid_ratio: float) -> int:
    return max(0, int(round(float(grid_size) * float(max(0.0, support_grid_ratio)))))


def build_variant_specs(
    *,
    support_grid_ratios: list[float],
    selected_variant_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_variant_names = selected_variant_names or []
    specs: list[dict[str, Any]] = []
    supported_by_name: dict[str, dict[str, Any]] = {}
    for ratio in support_grid_ratios:
        normalized_ratio = _normalize_support_grid_ratio(ratio)
        name = format_support_variant_name(normalized_ratio)
        if name in supported_by_name:
            continue
        supported_by_name[name] = {
            "name": name,
            "query_prefilter_mode": QUERY_PREFILTER_MODE_OFF,
            "support_grid_ratio": normalized_ratio,
        }

    if selected_variant_names:
        missing = [name for name in selected_variant_names if name not in supported_by_name]
        if missing:
            raise ValueError(
                f"Unsupported variants {missing}. Expected a subset of {sorted(supported_by_name.keys())}"
            )
        specs = [dict(supported_by_name[name]) for name in selected_variant_names]
    else:
        specs = [dict(supported_by_name[format_support_variant_name(ratio)]) for ratio in support_grid_ratios]

    return specs


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{time.time_ns()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _sync_cuda_if_needed(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _stdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _finite_percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _reason_bit_counts(reason_bits: np.ndarray | None) -> dict[str, int]:
    if reason_bits is None:
        return {}
    reason_bits = np.asarray(reason_bits, dtype=np.uint8).reshape(-1)
    return {
        f"bit_{bit_idx}": int(np.count_nonzero(reason_bits & np.uint8(1 << bit_idx)))
        for bit_idx in range(8)
    }


def _trim_track_mask(mask: np.ndarray, track_count: int) -> np.ndarray:
    return np.asarray(mask).astype(bool, copy=False).reshape(-1)[:track_count]


def _trim_traj(sample: dict[str, Any], track_count: int, frame_count: int) -> np.ndarray:
    return np.asarray(sample["traj_uvz"], dtype=np.float32)[:track_count, :frame_count]


def _trim_supervision_mask(
    sample: dict[str, Any],
    *,
    track_count: int,
    frame_count: int,
) -> np.ndarray | None:
    supervision_mask = sample.get("traj_supervision_mask")
    if supervision_mask is None:
        return None
    supervision_mask = np.asarray(supervision_mask).astype(bool, copy=False)
    if supervision_mask.ndim != 2:
        return None
    return supervision_mask[:track_count, :frame_count]


def _collect_masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"Expected matching shapes for values/mask, got {values.shape} vs {mask.shape}")
    if not np.any(mask):
        return np.asarray([], dtype=np.float32)
    return values[mask].astype(np.float32, copy=False)


def _trackwise_error_variance_mean(errors: np.ndarray, mask: np.ndarray) -> float | None:
    values: list[float] = []
    for track_errors, track_mask in zip(errors, mask, strict=False):
        masked_values = _collect_masked_values(track_errors, track_mask)
        if masked_values.size < 2:
            continue
        values.append(float(np.var(masked_values)))
    return _mean(values)


def _trackwise_endpoint_error_mean(errors: np.ndarray, mask: np.ndarray) -> float | None:
    values: list[float] = []
    for track_errors, track_mask in zip(errors, mask, strict=False):
        valid_indices = np.flatnonzero(track_mask)
        if valid_indices.size == 0:
            continue
        values.append(float(track_errors[valid_indices[-1]]))
    return _mean(values)


def _sample_world_tracks(
    sample: dict[str, Any],
    *,
    intrinsics_all: np.ndarray,
    extrinsics_all: np.ndarray,
    track_count: int,
    frame_count: int,
) -> np.ndarray | None:
    query_frame_index = int(sample["query_frame_index"])
    if query_frame_index < 0 or query_frame_index >= len(intrinsics_all) or query_frame_index >= len(extrinsics_all):
        return None
    return traj_uvz_to_world(
        _trim_traj(sample, track_count, frame_count),
        intrinsics_all[query_frame_index],
        extrinsics_all[query_frame_index],
    )


def summarize_sample(
    sample: dict[str, Any],
    *,
    sample_path: str | None = None,
) -> dict[str, Any]:
    traj_uvz = np.asarray(sample["traj_uvz"], dtype=np.float32)
    traj_valid_mask = np.asarray(sample["traj_valid_mask"]).astype(bool, copy=False)
    dense_query_count = int(sample.get("dense_query_count", traj_uvz.shape[0]))
    tracked_query_count = int(
        sample.get("tracked_query_count", int(np.isfinite(traj_uvz).any(axis=(1, 2)).sum()))
    )
    valid_track_count = int(traj_valid_mask.sum())
    supervision_count = sample.get("traj_supervision_count")
    supervision_count = (
        np.asarray(supervision_count, dtype=np.float32).reshape(-1)
        if supervision_count is not None
        else np.asarray([], dtype=np.float32)
    )
    tracked_ratio = (
        float(tracked_query_count / dense_query_count)
        if dense_query_count > 0
        else None
    )
    valid_ratio_dense = (
        float(valid_track_count / dense_query_count)
        if dense_query_count > 0
        else None
    )
    valid_ratio_tracked = (
        float(valid_track_count / tracked_query_count)
        if tracked_query_count > 0
        else None
    )
    return {
        "sample_path": sample_path,
        "query_frame_index": int(sample["query_frame_index"]),
        "dense_query_count": dense_query_count,
        "tracked_query_count": tracked_query_count,
        "valid_track_count": valid_track_count,
        "tracked_ratio": tracked_ratio,
        "valid_ratio_dense": valid_ratio_dense,
        "valid_ratio_tracked": valid_ratio_tracked,
        "support_grid_size": (
            int(sample["support_grid_size"])
            if sample.get("support_grid_size") is not None
            else None
        ),
        "traj_supervision_count_mean": (
            float(np.mean(supervision_count)) if supervision_count.size > 0 else None
        ),
        "traj_supervision_count_p50": _finite_percentile(supervision_count, 50.0),
        "traj_supervision_count_p90": _finite_percentile(supervision_count, 90.0),
        "traj_mask_reason_bit_counts": _reason_bit_counts(sample.get("traj_mask_reason_bits")),
    }


def summarize_case_samples(sample_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not sample_summaries:
        return {
            "sample_count": 0,
            "dense_query_count_mean": None,
            "tracked_query_count_mean": None,
            "valid_track_count_mean": None,
            "tracked_ratio_mean": None,
            "valid_ratio_dense_mean": None,
            "valid_ratio_tracked_mean": None,
            "support_grid_size_set": [],
            "traj_mask_reason_bit_counts_sum": {},
        }

    dense_counts = [int(item["dense_query_count"]) for item in sample_summaries.values()]
    tracked_counts = [int(item["tracked_query_count"]) for item in sample_summaries.values()]
    valid_counts = [int(item["valid_track_count"]) for item in sample_summaries.values()]
    tracked_ratios = [
        float(item["tracked_ratio"])
        for item in sample_summaries.values()
        if item["tracked_ratio"] is not None
    ]
    valid_ratio_dense = [
        float(item["valid_ratio_dense"])
        for item in sample_summaries.values()
        if item["valid_ratio_dense"] is not None
    ]
    valid_ratio_tracked = [
        float(item["valid_ratio_tracked"])
        for item in sample_summaries.values()
        if item["valid_ratio_tracked"] is not None
    ]
    support_grid_size_set = sorted(
        {
            int(item["support_grid_size"])
            for item in sample_summaries.values()
            if item["support_grid_size"] is not None
        }
    )
    reason_bit_counts_sum: dict[str, int] = {}
    for item in sample_summaries.values():
        for key, value in item["traj_mask_reason_bit_counts"].items():
            reason_bit_counts_sum[key] = int(reason_bit_counts_sum.get(key, 0) + int(value))
    return {
        "sample_count": int(len(sample_summaries)),
        "dense_query_count_mean": _mean([float(value) for value in dense_counts]),
        "tracked_query_count_mean": _mean([float(value) for value in tracked_counts]),
        "valid_track_count_mean": _mean([float(value) for value in valid_counts]),
        "tracked_ratio_mean": _mean(tracked_ratios),
        "valid_ratio_dense_mean": _mean(valid_ratio_dense),
        "valid_ratio_tracked_mean": _mean(valid_ratio_tracked),
        "support_grid_size_set": support_grid_size_set,
        "traj_mask_reason_bit_counts_sum": reason_bit_counts_sum,
    }


def collect_sample_summaries(
    run_output_root: Path,
    *,
    camera_name: str,
) -> dict[str, dict[str, Any]]:
    samples_dir = run_output_root / camera_name / "samples"
    if not samples_dir.is_dir():
        return {}
    sample_summaries: dict[str, dict[str, Any]] = {}
    for sample_path in sorted(samples_dir.glob("*.npz")):
        sample = normalize_sample_data(sample_path)
        sample_summary = summarize_sample(sample, sample_path=str(sample_path))
        sample_summaries[str(sample_summary["query_frame_index"])] = sample_summary
    return sample_summaries


def load_camera_samples(
    run_output_root: Path,
    *,
    camera_name: str,
) -> dict[int, dict[str, Any]]:
    samples_dir = run_output_root / camera_name / "samples"
    if not samples_dir.is_dir():
        return {}
    samples: dict[int, dict[str, Any]] = {}
    for sample_path in sorted(samples_dir.glob("*.npz")):
        sample = normalize_sample_data(sample_path)
        samples[int(sample["query_frame_index"])] = sample
    return samples


def compare_sample_pair(
    baseline_sample: dict[str, Any],
    variant_sample: dict[str, Any],
    *,
    baseline_camera_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    variant_camera_arrays: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    baseline_summary = summarize_sample(baseline_sample)
    variant_summary = summarize_sample(variant_sample)

    baseline_valid_mask_full = np.asarray(baseline_sample["traj_valid_mask"]).astype(bool, copy=False)
    variant_valid_mask_full = np.asarray(variant_sample["traj_valid_mask"]).astype(bool, copy=False)
    track_count = min(len(baseline_valid_mask_full), len(variant_valid_mask_full))
    frame_count = min(
        np.asarray(baseline_sample["traj_uvz"]).shape[1],
        np.asarray(variant_sample["traj_uvz"]).shape[1],
    )

    dense_count_match = baseline_valid_mask_full.shape == variant_valid_mask_full.shape
    baseline_valid_mask = _trim_track_mask(baseline_valid_mask_full, track_count)
    variant_valid_mask = _trim_track_mask(variant_valid_mask_full, track_count)

    union_mask = baseline_valid_mask | variant_valid_mask
    intersection_mask = baseline_valid_mask & variant_valid_mask
    traj_valid_mask_jaccard = (
        float(intersection_mask.sum() / union_mask.sum())
        if np.any(union_mask)
        else 1.0
    )

    traj_uvz_mae = None
    traj_2d_l2_mean = None
    traj_depth_abs_mean = None
    traj_world_l2_mean = None
    traj_world_l2_p95 = None
    traj_world_error_var_mean = None
    traj_world_endpoint_l2_mean = None
    traj_world_step_delta_l2_mean = None
    traj_world_step_delta_l2_p95 = None
    common_valid_step_count = 0
    common_valid_track_count = int(intersection_mask.sum())

    if common_valid_track_count > 0 and frame_count > 0:
        baseline_traj = _trim_traj(baseline_sample, track_count, frame_count)[intersection_mask]
        variant_traj = _trim_traj(variant_sample, track_count, frame_count)[intersection_mask]
        common_step_mask = (
            np.isfinite(baseline_traj).all(axis=-1)
            & np.isfinite(variant_traj).all(axis=-1)
        )
        baseline_supervision_mask = _trim_supervision_mask(
            baseline_sample,
            track_count=track_count,
            frame_count=frame_count,
        )
        if baseline_supervision_mask is not None:
            common_step_mask &= baseline_supervision_mask[intersection_mask]
        variant_supervision_mask = _trim_supervision_mask(
            variant_sample,
            track_count=track_count,
            frame_count=frame_count,
        )
        if variant_supervision_mask is not None:
            common_step_mask &= variant_supervision_mask[intersection_mask]

        common_valid_step_count = int(common_step_mask.sum())
        if common_valid_step_count > 0:
            traj_diff = variant_traj - baseline_traj
            finite_flat_mask = np.broadcast_to(common_step_mask[..., None], traj_diff.shape)
            traj_uvz_mae = float(np.abs(traj_diff[finite_flat_mask]).mean())

            traj_2d_error = np.linalg.norm(traj_diff[..., :2], axis=-1)
            traj_2d_values = _collect_masked_values(traj_2d_error, common_step_mask)
            if traj_2d_values.size > 0:
                traj_2d_l2_mean = float(traj_2d_values.mean())

            traj_depth_values = _collect_masked_values(np.abs(traj_diff[..., 2]), common_step_mask)
            if traj_depth_values.size > 0:
                traj_depth_abs_mean = float(traj_depth_values.mean())

            if baseline_camera_arrays is not None and variant_camera_arrays is not None:
                baseline_world = _sample_world_tracks(
                    baseline_sample,
                    intrinsics_all=baseline_camera_arrays[0],
                    extrinsics_all=baseline_camera_arrays[1],
                    track_count=track_count,
                    frame_count=frame_count,
                )
                variant_world = _sample_world_tracks(
                    variant_sample,
                    intrinsics_all=variant_camera_arrays[0],
                    extrinsics_all=variant_camera_arrays[1],
                    track_count=track_count,
                    frame_count=frame_count,
                )
                if baseline_world is not None and variant_world is not None:
                    baseline_world = baseline_world[intersection_mask]
                    variant_world = variant_world[intersection_mask]
                    world_error = np.linalg.norm(variant_world - baseline_world, axis=-1)
                    world_step_mask = common_step_mask & np.isfinite(world_error)
                    world_values = _collect_masked_values(world_error, world_step_mask)
                    if world_values.size > 0:
                        traj_world_l2_mean = float(world_values.mean())
                        traj_world_l2_p95 = _finite_percentile(world_values, 95.0)
                        traj_world_error_var_mean = _trackwise_error_variance_mean(world_error, world_step_mask)
                        traj_world_endpoint_l2_mean = _trackwise_endpoint_error_mean(world_error, world_step_mask)

                    if frame_count >= 2:
                        baseline_step = np.diff(baseline_world, axis=1)
                        variant_step = np.diff(variant_world, axis=1)
                        step_mask = world_step_mask[:, 1:] & world_step_mask[:, :-1]
                        step_mask &= np.isfinite(baseline_step).all(axis=-1)
                        step_mask &= np.isfinite(variant_step).all(axis=-1)
                        step_error = np.linalg.norm(variant_step - baseline_step, axis=-1)
                        step_values = _collect_masked_values(step_error, step_mask)
                        if step_values.size > 0:
                            traj_world_step_delta_l2_mean = float(step_values.mean())
                            traj_world_step_delta_l2_p95 = _finite_percentile(step_values, 95.0)

    return {
        "query_frame_index": int(baseline_sample["query_frame_index"]),
        "dense_count_match": bool(dense_count_match),
        "baseline_dense_query_count": int(baseline_summary["dense_query_count"]),
        "variant_dense_query_count": int(variant_summary["dense_query_count"]),
        "baseline_tracked_query_count": int(baseline_summary["tracked_query_count"]),
        "variant_tracked_query_count": int(variant_summary["tracked_query_count"]),
        "tracked_query_count_delta": int(
            variant_summary["tracked_query_count"] - baseline_summary["tracked_query_count"]
        ),
        "baseline_valid_track_count": int(baseline_summary["valid_track_count"]),
        "variant_valid_track_count": int(variant_summary["valid_track_count"]),
        "valid_track_count_delta": int(
            variant_summary["valid_track_count"] - baseline_summary["valid_track_count"]
        ),
        "traj_valid_mask_jaccard": float(traj_valid_mask_jaccard),
        "common_valid_track_count": common_valid_track_count,
        "common_valid_step_count": common_valid_step_count,
        "traj_uvz_mae": traj_uvz_mae,
        "traj_2d_l2_mean": traj_2d_l2_mean,
        "traj_depth_abs_mean": traj_depth_abs_mean,
        "traj_world_l2_mean": traj_world_l2_mean,
        "traj_world_l2_p95": traj_world_l2_p95,
        "traj_world_error_var_mean": traj_world_error_var_mean,
        "traj_world_endpoint_l2_mean": traj_world_endpoint_l2_mean,
        "traj_world_step_delta_l2_mean": traj_world_step_delta_l2_mean,
        "traj_world_step_delta_l2_p95": traj_world_step_delta_l2_p95,
    }


def compare_camera_outputs(
    baseline_output_dir: Path,
    variant_output_dir: Path,
    *,
    camera_name: str,
) -> dict[str, Any]:
    baseline_samples = load_camera_samples(baseline_output_dir, camera_name=camera_name)
    variant_samples = load_camera_samples(variant_output_dir, camera_name=camera_name)
    baseline_query_frames = sorted(baseline_samples.keys())
    variant_query_frames = sorted(variant_samples.keys())
    common_query_frames = sorted(set(baseline_query_frames) & set(variant_query_frames))

    per_sample: dict[str, dict[str, Any]] = {}
    tracked_deltas: list[float] = []
    valid_deltas: list[float] = []
    jaccard_values: list[float] = []
    common_valid_counts: list[float] = []
    common_valid_step_counts: list[float] = []
    traj_uvz_mae_values: list[float] = []
    traj_2d_l2_values: list[float] = []
    traj_depth_abs_values: list[float] = []
    traj_world_l2_mean_values: list[float] = []
    traj_world_l2_p95_values: list[float] = []
    traj_world_error_var_values: list[float] = []
    traj_world_endpoint_values: list[float] = []
    traj_world_step_delta_mean_values: list[float] = []
    traj_world_step_delta_p95_values: list[float] = []

    baseline_camera_dir = baseline_output_dir / camera_name
    variant_camera_dir = variant_output_dir / camera_name
    with SceneReader(baseline_camera_dir) as baseline_reader, SceneReader(variant_camera_dir) as variant_reader:
        baseline_camera_arrays = baseline_reader.get_camera_arrays()
        variant_camera_arrays = variant_reader.get_camera_arrays()

        for query_frame in common_query_frames:
            comparison = compare_sample_pair(
                baseline_samples[query_frame],
                variant_samples[query_frame],
                baseline_camera_arrays=baseline_camera_arrays,
                variant_camera_arrays=variant_camera_arrays,
            )
            per_sample[str(query_frame)] = comparison
            tracked_deltas.append(float(comparison["tracked_query_count_delta"]))
            valid_deltas.append(float(comparison["valid_track_count_delta"]))
            jaccard_values.append(float(comparison["traj_valid_mask_jaccard"]))
            common_valid_counts.append(float(comparison["common_valid_track_count"]))
            common_valid_step_counts.append(float(comparison["common_valid_step_count"]))
            if comparison["traj_uvz_mae"] is not None:
                traj_uvz_mae_values.append(float(comparison["traj_uvz_mae"]))
            if comparison["traj_2d_l2_mean"] is not None:
                traj_2d_l2_values.append(float(comparison["traj_2d_l2_mean"]))
            if comparison["traj_depth_abs_mean"] is not None:
                traj_depth_abs_values.append(float(comparison["traj_depth_abs_mean"]))
            if comparison["traj_world_l2_mean"] is not None:
                traj_world_l2_mean_values.append(float(comparison["traj_world_l2_mean"]))
            if comparison["traj_world_l2_p95"] is not None:
                traj_world_l2_p95_values.append(float(comparison["traj_world_l2_p95"]))
            if comparison["traj_world_error_var_mean"] is not None:
                traj_world_error_var_values.append(float(comparison["traj_world_error_var_mean"]))
            if comparison["traj_world_endpoint_l2_mean"] is not None:
                traj_world_endpoint_values.append(float(comparison["traj_world_endpoint_l2_mean"]))
            if comparison["traj_world_step_delta_l2_mean"] is not None:
                traj_world_step_delta_mean_values.append(float(comparison["traj_world_step_delta_l2_mean"]))
            if comparison["traj_world_step_delta_l2_p95"] is not None:
                traj_world_step_delta_p95_values.append(float(comparison["traj_world_step_delta_l2_p95"]))

    return {
        "camera_name": camera_name,
        "baseline_query_frames": baseline_query_frames,
        "variant_query_frames": variant_query_frames,
        "common_query_frames": common_query_frames,
        "baseline_only_query_frames": sorted(set(baseline_query_frames) - set(variant_query_frames)),
        "variant_only_query_frames": sorted(set(variant_query_frames) - set(baseline_query_frames)),
        "per_sample": per_sample,
        "aggregates": {
            "common_query_frame_count": int(len(common_query_frames)),
            "tracked_query_count_delta_mean": _mean(tracked_deltas),
            "valid_track_count_delta_mean": _mean(valid_deltas),
            "traj_valid_mask_jaccard_mean": _mean(jaccard_values),
            "common_valid_track_count_mean": _mean(common_valid_counts),
            "common_valid_step_count_mean": _mean(common_valid_step_counts),
            "traj_uvz_mae_mean": _mean(traj_uvz_mae_values),
            "traj_2d_l2_mean": _mean(traj_2d_l2_values),
            "traj_depth_abs_mean": _mean(traj_depth_abs_values),
            "traj_world_l2_mean": _mean(traj_world_l2_mean_values),
            "traj_world_l2_p95": _mean(traj_world_l2_p95_values),
            "traj_world_error_var_mean": _mean(traj_world_error_var_values),
            "traj_world_endpoint_l2_mean": _mean(traj_world_endpoint_values),
            "traj_world_step_delta_l2_mean": _mean(traj_world_step_delta_mean_values),
            "traj_world_step_delta_l2_p95": _mean(traj_world_step_delta_p95_values),
        },
    }


def aggregate_profile_stats(run_records: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    keys = sorted(
        {
            key
            for record in run_records
            for key in record.get("profile_stats", {}).keys()
        }
    )
    aggregated: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [
            float(record["profile_stats"][key])
            for record in run_records
            if key in record.get("profile_stats", {})
        ]
        aggregated[key] = {
            "mean": _mean(values),
            "stdev": _stdev(values),
        }
    return aggregated


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.episode_dir is None:
        raise ValueError("--episode-dir is required in worker mode")
    if args.camera_name is None:
        raise ValueError("--camera-name is required in worker mode")
    if args.variant_name is None:
        raise ValueError("--variant-name is required in worker mode")
    if args.run_label is None:
        raise ValueError("--run-label is required in worker mode")
    if args.query_frame_schedule_path is None:
        raise ValueError("--query-frame-schedule-path is required in worker mode")
    if args.result_json is None:
        raise ValueError("--result-json is required in worker mode")

    args.episode_dir = args.episode_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    args.query_frame_schedule_path = args.query_frame_schedule_path.resolve()
    args.result_json = args.result_json.resolve()

    variant_spec = build_variant_specs(
        support_grid_ratios=parse_support_grid_ratios(args.support_grid_ratios),
        selected_variant_names=[args.variant_name],
    )[0]

    import importlib

    infer_module = importlib.import_module("scripts.batch_inference.infer")
    torch = importlib.import_module("torch")

    profile = resolve_traj_filter_profile(args.camera_name, args.traj_filter_profile)
    infer_args = argparse.Namespace(
        video_path=str(args.episode_dir / "rgb" / args.camera_name),
        depth_path=str(args.episode_dir / "depth" / args.camera_name),
        mask_dir=None,
        checkpoint=str(args.checkpoint),
        depth_pose_method="external",
        external_extr_mode=args.external_extr_mode,
        external_geom_npz=str(args.episode_dir / args.external_geom_name),
        camera_name=args.camera_name,
        query_frame_schedule_path=str(args.query_frame_schedule_path),
        device=args.device,
        num_iters=args.num_iters,
        fps=args.fps,
        out_dir=str(args.output_root),
        video_name=args.camera_name,
        max_num_frames=args.max_num_frames,
        save_video=False,
        output_layout="v2",
        scene_storage_mode="source_ref",
        save_visibility=False,
        horizon=16,
        batch_process=False,
        skip_existing=False,
        use_all_trajectories=True,
        frame_drop_rate=1,
        scan_depth=2,
        future_len=args.future_len,
        max_frames_per_video=args.max_num_frames,
        grid_size=args.grid_size,
        filter_level=args.filter_level,
        traj_filter_profile=profile,
        min_valid_frames=None,
        visibility_threshold=None,
        min_depth=0.01,
        max_depth=10.0,
        boundary_margin=None,
        depth_change_threshold=None,
        query_prefilter_mode=QUERY_PREFILTER_MODE_OFF,
        query_prefilter_wrist_rank_keep_ratio=0.30,
        support_grid_ratio=float(variant_spec["support_grid_ratio"]),
        collect_profile_stats=True,
    )

    model_depth_pose = infer_module.video_depth_pose_dict[infer_args.depth_pose_method](infer_args)
    model_3dtracker = infer_module.load_model(str(args.checkpoint)).to(args.device)

    raw_runs: list[dict[str, Any]] = []
    measured_runs: list[dict[str, Any]] = []
    representative_output_dir: str | None = None
    representative_sample_summaries: dict[str, dict[str, Any]] | None = None

    total_runs = int(args.warmup_runs) + int(args.benchmark_runs)
    artifacts_root = args.output_root / "artifacts" / args.run_label
    for run_idx in range(total_runs):
        run_name = f"run_{run_idx:02d}"
        run_output_root = artifacts_root / run_name
        if run_output_root.exists():
            shutil.rmtree(run_output_root)
        run_output_root.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _sync_cuda_if_needed(torch, args.device)
        process_start = time.perf_counter()
        result = infer_module.process_single_video(
            str(args.episode_dir / "rgb" / args.camera_name),
            str(args.episode_dir / "depth" / args.camera_name),
            infer_args,
            model_3dtracker,
            model_depth_pose,
            args.camera_name,
            str(run_output_root),
        )
        _sync_cuda_if_needed(torch, args.device)
        process_seconds = time.perf_counter() - process_start

        save_start = time.perf_counter()
        infer_module.save_structured_data(
            video_name=args.camera_name,
            output_dir=str(run_output_root),
            video_tensor=result["video_tensor"],
            depths=result["depths"],
            coords=result["coords"],
            visibs=result["visibs"],
            intrinsics=result["intrinsics"],
            extrinsics=result["extrinsics"],
            query_points_per_frame=result["query_points_per_frame"],
            original_filenames=result["original_filenames"],
            query_frame_results=result.get("query_frame_results"),
            future_len=infer_args.future_len,
            grid_size=infer_args.grid_size,
            filter_args=infer_args,
            full_video_tensor=result["full_video_tensor"],
            full_depths=result["full_depths"],
            full_intrinsics=result["full_intrinsics"],
            full_extrinsics=result["full_extrinsics"],
            depth_conf=result["depth_conf"],
            video_source_path=str(args.episode_dir / "rgb" / args.camera_name),
            depth_source_path=str(args.episode_dir / "depth" / args.camera_name),
            source_frame_indices=result["source_frame_indices"],
            query_frame_metadata=result.get("query_frame_metadata"),
        )
        _sync_cuda_if_needed(torch, args.device)
        save_seconds = time.perf_counter() - save_start

        sample_summaries = collect_sample_summaries(run_output_root, camera_name=args.camera_name)
        query_frame_results = result.get("query_frame_results") or {}
        tracked_query_counts = [
            int(frame_data["tracked_query_count"])
            for frame_data in query_frame_results.values()
        ]
        dense_query_counts = [
            int(frame_data["dense_query_count"])
            for frame_data in query_frame_results.values()
        ]
        support_grid_sizes = [
            int(frame_data["support_grid_size"])
            for frame_data in query_frame_results.values()
            if frame_data.get("support_grid_size") is not None
        ]
        effective_support_query_counts = [
            int(frame_data.get("effective_support_query_count", 0))
            for frame_data in query_frame_results.values()
        ]
        profile_stats = {
            key: float(value)
            for key, value in (result.get("profile_stats") or {}).items()
            if value is not None
        }
        run_record = {
            "run_index": int(run_idx),
            "warmup": bool(run_idx < args.warmup_runs),
            "process_seconds": float(process_seconds),
            "save_seconds": float(save_seconds),
            "total_seconds": float(process_seconds + save_seconds),
            "query_frame_count": int(len(query_frame_results)),
            "saved_sample_count": int(len(sample_summaries)),
            "frame_count": int(result["full_video_tensor"].shape[0]),
            "dense_query_count_mean": _mean([float(value) for value in dense_query_counts]),
            "tracked_query_count_mean": _mean([float(value) for value in tracked_query_counts]),
            "support_grid_sizes": support_grid_sizes,
            "support_grid_size_mean": _mean([float(value) for value in support_grid_sizes]),
            "effective_support_query_counts": effective_support_query_counts,
            "effective_support_query_count_mean": _mean(
                [float(value) for value in effective_support_query_counts]
            ),
            "profile_stats": profile_stats,
            "output_dir": str(run_output_root),
        }
        raw_runs.append(run_record)
        if run_idx >= args.warmup_runs:
            measured_runs.append(run_record)

        keep_run_outputs = bool(args.keep_outputs)
        if run_idx >= args.warmup_runs and representative_output_dir is None:
            representative_output_dir = str(run_output_root)
            representative_sample_summaries = sample_summaries
            keep_run_outputs = True

        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not keep_run_outputs:
            shutil.rmtree(run_output_root)

    del model_3dtracker
    del model_depth_pose
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    process_values = [run["process_seconds"] for run in measured_runs]
    save_values = [run["save_seconds"] for run in measured_runs]
    total_values = [run["total_seconds"] for run in measured_runs]
    query_frame_count_values = [float(run["query_frame_count"]) for run in measured_runs]
    saved_sample_count_values = [float(run["saved_sample_count"]) for run in measured_runs]
    tracked_query_count_values = [
        float(run["tracked_query_count_mean"])
        for run in measured_runs
        if run["tracked_query_count_mean"] is not None
    ]
    dense_query_count_values = [
        float(run["dense_query_count_mean"])
        for run in measured_runs
        if run["dense_query_count_mean"] is not None
    ]
    support_grid_size_values = [
        float(value)
        for run in measured_runs
        for value in run.get("support_grid_sizes", [])
    ]
    effective_support_query_count_values = [
        float(value)
        for run in measured_runs
        for value in run.get("effective_support_query_counts", [])
    ]
    representative_sample_summaries = representative_sample_summaries or {}
    requested_support_grid_size = resolve_requested_support_grid_size(
        args.grid_size,
        float(variant_spec["support_grid_ratio"]),
    )
    summary = {
        "variant_name": args.variant_name,
        "variant_config": {
            **variant_spec,
            "requested_support_grid_size": int(requested_support_grid_size),
            "requested_support_query_capacity": int(requested_support_grid_size * requested_support_grid_size),
        },
        "camera_name": args.camera_name,
        "traj_filter_profile": profile,
        "schedule_path": str(args.query_frame_schedule_path),
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "keep_outputs": bool(args.keep_outputs),
        "raw_runs": raw_runs,
        "measured_runs": measured_runs,
        "aggregates": {
            "process_seconds_mean": _mean(process_values),
            "process_seconds_stdev": _stdev(process_values),
            "save_seconds_mean": _mean(save_values),
            "save_seconds_stdev": _stdev(save_values),
            "total_seconds_mean": _mean(total_values),
            "total_seconds_stdev": _stdev(total_values),
            "query_frame_count_mean": _mean(query_frame_count_values),
            "saved_sample_count_mean": _mean(saved_sample_count_values),
            "tracked_query_count_mean": _mean(tracked_query_count_values),
            "dense_query_count_mean": _mean(dense_query_count_values),
            "support_grid_size_mean": _mean(support_grid_size_values),
            "support_grid_size_p50": _finite_percentile(np.asarray(support_grid_size_values, dtype=np.float32), 50.0),
            "support_grid_size_p90": _finite_percentile(np.asarray(support_grid_size_values, dtype=np.float32), 90.0),
            "effective_support_query_count_mean": _mean(effective_support_query_count_values),
            "effective_support_query_count_p50": _finite_percentile(
                np.asarray(effective_support_query_count_values, dtype=np.float32), 50.0
            ),
            "effective_support_query_count_p90": _finite_percentile(
                np.asarray(effective_support_query_count_values, dtype=np.float32), 90.0
            ),
            "requested_support_grid_size": int(requested_support_grid_size),
            "requested_support_query_capacity": int(requested_support_grid_size * requested_support_grid_size),
        },
        "profile_aggregates": aggregate_profile_stats(measured_runs),
        "representative_output_dir": representative_output_dir,
        "representative_sample_overview": summarize_case_samples(representative_sample_summaries),
        "representative_sample_summaries": representative_sample_summaries,
    }
    _atomic_write_json(args.result_json, summary)
    return summary


def run_visual_verification(
    *,
    summary_by_key: dict[tuple[str, str], dict[str, Any]],
    variant_specs: list[dict[str, Any]],
    camera_names: list[str],
    output_root: Path,
) -> list[dict[str, Any]]:
    verification_results: list[dict[str, Any]] = []
    verify_script_path = CURRENT_REPO_ROOT / "scripts" / "visualization" / "verify_episode_trajectory_outputs.py"
    for camera_name in camera_names:
        variant_query_frame_sets: list[set[int]] = []
        for variant_spec in variant_specs:
            case = summary_by_key[(variant_spec["name"], camera_name)]
            representative_output_dir = case.get("representative_output_dir")
            if representative_output_dir is None:
                variant_query_frame_sets = []
                break
            samples = load_camera_samples(Path(representative_output_dir), camera_name=camera_name)
            variant_query_frame_sets.append(set(samples.keys()))
        if not variant_query_frame_sets:
            continue
        common_query_frames = sorted(set.intersection(*variant_query_frame_sets))
        if not common_query_frames:
            verification_results.append(
                {
                    "camera_name": camera_name,
                    "query_frame": None,
                    "variants": [],
                }
            )
            continue
        query_frame = int(common_query_frames[0])
        variant_outputs: list[dict[str, Any]] = []
        for variant_spec in variant_specs:
            case = summary_by_key[(variant_spec["name"], camera_name)]
            representative_output_dir = Path(case["representative_output_dir"])
            variant_output_dir = output_root / "visual_verification" / camera_name / variant_spec["name"]
            cmd = [
                sys.executable,
                str(verify_script_path),
                "--episode_dir",
                str(representative_output_dir),
                "--trajectory_dirname",
                ".",
                "--camera_names",
                camera_name,
                "--query_frames",
                str(query_frame),
                "--output_dir",
                str(variant_output_dir),
            ]
            subprocess.run(cmd, check=True)
            variant_outputs.append(
                {
                    "variant_name": variant_spec["name"],
                    "output_dir": str(variant_output_dir),
                    "summary_path": str(variant_output_dir / "summary.json"),
                }
            )
        verification_results.append(
            {
                "camera_name": camera_name,
                "query_frame": query_frame,
                "variants": variant_outputs,
            }
        )
    return verification_results


def build_summary_payload(
    *,
    args: argparse.Namespace,
    schedule_path: Path,
    variant_specs: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
    verification_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_by_key = {
        (case["variant_name"], case["camera_name"]): case
        for case in case_results
    }
    variant_rows: list[dict[str, Any]] = []
    pairwise_comparisons: list[dict[str, Any]] = []
    for camera_name in parse_camera_names(args.camera_names):
        baseline_case = summary_by_key[("baseline", camera_name)]
        baseline_prepare_inputs = _to_float_or_none(
            baseline_case["profile_aggregates"].get("prepare_inputs_seconds", {}).get("mean")
        )
        baseline_tracker_forward = _to_float_or_none(
            baseline_case["profile_aggregates"].get("tracker_model_forward_seconds", {}).get("mean")
        )
        baseline_tracker_total = _to_float_or_none(
            baseline_case["profile_aggregates"].get("tracker_inference_total_seconds", {}).get("mean")
        )

        for variant_spec in variant_specs:
            case = summary_by_key[(variant_spec["name"], camera_name)]
            variant_rows.append(
                {
                    "camera_name": camera_name,
                    "variant_name": variant_spec["name"],
                    "traj_filter_profile": case["traj_filter_profile"],
                    "variant_config": case["variant_config"],
                    "aggregates": case["aggregates"],
                    "profile_aggregates": case["profile_aggregates"],
                    "representative_sample_overview": case["representative_sample_overview"],
                    "representative_output_dir": case["representative_output_dir"],
                }
            )
            if variant_spec["name"] == "baseline":
                continue
            sample_diff = compare_camera_outputs(
                Path(baseline_case["representative_output_dir"]),
                Path(case["representative_output_dir"]),
                camera_name=camera_name,
            )
            variant_prepare_inputs = _to_float_or_none(
                case["profile_aggregates"].get("prepare_inputs_seconds", {}).get("mean")
            )
            variant_tracker_forward = _to_float_or_none(
                case["profile_aggregates"].get("tracker_model_forward_seconds", {}).get("mean")
            )
            variant_tracker_total = _to_float_or_none(
                case["profile_aggregates"].get("tracker_inference_total_seconds", {}).get("mean")
            )
            pairwise_comparisons.append(
                {
                    "camera_name": camera_name,
                    "variant_name": variant_spec["name"],
                    "traj_filter_profile": case["traj_filter_profile"],
                    "variant_config": case["variant_config"],
                    "process_speedup_vs_baseline": _safe_ratio(
                        baseline_case["aggregates"]["process_seconds_mean"],
                        case["aggregates"]["process_seconds_mean"],
                    ),
                    "save_speedup_vs_baseline": _safe_ratio(
                        baseline_case["aggregates"]["save_seconds_mean"],
                        case["aggregates"]["save_seconds_mean"],
                    ),
                    "total_speedup_vs_baseline": _safe_ratio(
                        baseline_case["aggregates"]["total_seconds_mean"],
                        case["aggregates"]["total_seconds_mean"],
                    ),
                    "prepare_inputs_speedup_vs_baseline": _safe_ratio(
                        baseline_prepare_inputs,
                        variant_prepare_inputs,
                    ),
                    "tracker_inference_speedup_vs_baseline": _safe_ratio(
                        baseline_tracker_total,
                        variant_tracker_total,
                    ),
                    "tracker_forward_speedup_vs_baseline": _safe_ratio(
                        baseline_tracker_forward,
                        variant_tracker_forward,
                    ),
                    "baseline": baseline_case["aggregates"],
                    "variant": case["aggregates"],
                    "sample_diff": sample_diff,
                }
            )

    return {
        "episode_dir": str(args.episode_dir),
        "camera_names": parse_camera_names(args.camera_names),
        "support_grid_ratios": parse_support_grid_ratios(args.support_grid_ratios),
        "variant_specs": variant_specs,
        "schedule_path": str(schedule_path),
        "current_repo_root": str(CURRENT_REPO_ROOT.resolve()),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "fps": int(args.fps),
        "max_num_frames": int(args.max_num_frames),
        "future_len": int(args.future_len),
        "grid_size": int(args.grid_size),
        "filter_level": args.filter_level,
        "traj_filter_profile": args.traj_filter_profile,
        "keyframes_per_sec_min": int(args.keyframes_per_sec_min),
        "keyframes_per_sec_max": int(args.keyframes_per_sec_max),
        "keep_outputs": bool(args.keep_outputs),
        "run_visual_verification": bool(args.run_visual_verification),
        "case_results": case_results,
        "variant_rows": variant_rows,
        "pairwise_comparisons": pairwise_comparisons,
        "verification_results": verification_results,
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _find_pareto_candidates(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    pairwise_by_key = {
        (item["camera_name"], item["variant_name"]): item
        for item in summary["pairwise_comparisons"]
    }
    candidates_by_camera: dict[str, list[dict[str, Any]]] = {}
    for camera_name in summary["camera_names"]:
        camera_rows = [row for row in summary["variant_rows"] if row["camera_name"] == camera_name]
        candidate_rows: list[dict[str, Any]] = []
        comparable_rows: list[dict[str, Any]] = []
        for row in camera_rows:
            variant_name = row["variant_name"]
            if variant_name == "baseline":
                metrics = {
                    "traj_valid_mask_jaccard_mean": 1.0,
                    "traj_world_l2_mean": 0.0,
                }
            else:
                metrics = pairwise_by_key[(camera_name, variant_name)]["sample_diff"]["aggregates"]
            comparable_rows.append(
                {
                    "variant_name": variant_name,
                    "support_grid_ratio": row["variant_config"]["support_grid_ratio"],
                    "total_seconds_mean": _to_float_or_none(row["aggregates"]["total_seconds_mean"]),
                    "traj_valid_mask_jaccard_mean": _to_float_or_none(metrics["traj_valid_mask_jaccard_mean"]),
                    "traj_world_l2_mean": _to_float_or_none(metrics["traj_world_l2_mean"]),
                }
            )

        for row in comparable_rows:
            dominated = False
            row_total = row["total_seconds_mean"]
            row_jaccard = row["traj_valid_mask_jaccard_mean"]
            row_world = row["traj_world_l2_mean"]
            if row_total is None or row_jaccard is None or row_world is None:
                candidate_rows.append(row)
                continue
            for other in comparable_rows:
                if other["variant_name"] == row["variant_name"]:
                    continue
                other_total = other["total_seconds_mean"]
                other_jaccard = other["traj_valid_mask_jaccard_mean"]
                other_world = other["traj_world_l2_mean"]
                if other_total is None or other_jaccard is None or other_world is None:
                    continue
                dominates = (
                    other_total <= row_total
                    and other_jaccard >= row_jaccard
                    and other_world <= row_world
                    and (
                        other_total < row_total
                        or other_jaccard > row_jaccard
                        or other_world < row_world
                    )
                )
                if dominates:
                    dominated = True
                    break
            if not dominated:
                candidate_rows.append(row)
        candidates_by_camera[camera_name] = candidate_rows
    return candidates_by_camera


def write_summary_markdown(summary: dict[str, Any], summary_path: Path) -> None:
    lines = [
        "# Inference Support Sweep Benchmark Summary",
        "",
        f"- Episode: `{summary['episode_dir']}`",
        f"- Cameras: `{','.join(summary['camera_names'])}`",
        f"- Support ratios: `{','.join(str(item) for item in summary['support_grid_ratios'])}`",
        f"- Variants: `{','.join(item['name'] for item in summary['variant_specs'])}`",
        f"- Device: `{summary['device']}`",
        f"- Schedule: `{summary['schedule_path']}`",
        f"- Measured runs: `{summary['benchmark_runs']}`",
        f"- Warmup runs: `{summary['warmup_runs']}`",
        "",
        "## Timing",
        "",
        "| Camera | Variant | Ratio | Support Grid | Effective Support Count | Process (s) | Save (s) | Total (s) | Prepare Inputs (s) | Tracker Forward (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["variant_rows"]:
        profile_aggregates = row["profile_aggregates"]
        aggregates = row["aggregates"]
        prepare_inputs_mean = _to_float_or_none(
            profile_aggregates.get("prepare_inputs_seconds", {}).get("mean")
        )
        tracker_forward_mean = _to_float_or_none(
            profile_aggregates.get("tracker_model_forward_seconds", {}).get("mean")
        )
        lines.append(
            "| {camera} | {variant} | {ratio} | {support_grid} | {effective_support} | {process} | {save} | {total} | {prepare_inputs} | {tracker_forward} |".format(
                camera=row["camera_name"],
                variant=row["variant_name"],
                ratio=_format_float(row["variant_config"]["support_grid_ratio"], digits=2),
                support_grid=_format_float(aggregates["requested_support_grid_size"], digits=0),
                effective_support=_format_float(aggregates["effective_support_query_count_mean"], digits=1),
                process=_format_float(aggregates["process_seconds_mean"]),
                save=_format_float(aggregates["save_seconds_mean"]),
                total=_format_float(aggregates["total_seconds_mean"]),
                prepare_inputs=_format_float(prepare_inputs_mean),
                tracker_forward=_format_float(tracker_forward_mean),
            )
        )

    lines.extend(
        [
            "",
            "## Quality vs Baseline",
            "",
            "| Camera | Variant | Ratio | Process Speedup | Total Speedup | Forward Speedup | Valid Delta | Mask Jaccard | World L2 Mean | World L2 P95 | World Error Var Mean | Step Delta Mean | Endpoint Error |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in summary["pairwise_comparisons"]:
        diff_aggregates = comparison["sample_diff"]["aggregates"]
        lines.append(
            "| {camera} | {variant} | {ratio} | {process_speedup}x | {total_speedup}x | {forward_speedup}x | {valid_delta} | {jaccard} | {world_l2_mean} | {world_l2_p95} | {world_error_var} | {step_delta} | {endpoint_error} |".format(
                camera=comparison["camera_name"],
                variant=comparison["variant_name"],
                ratio=_format_float(comparison["variant_config"]["support_grid_ratio"], digits=2),
                process_speedup=_format_float(comparison["process_speedup_vs_baseline"]),
                total_speedup=_format_float(comparison["total_speedup_vs_baseline"]),
                forward_speedup=_format_float(comparison["tracker_forward_speedup_vs_baseline"]),
                valid_delta=_format_float(diff_aggregates["valid_track_count_delta_mean"]),
                jaccard=_format_float(diff_aggregates["traj_valid_mask_jaccard_mean"]),
                world_l2_mean=_format_float(diff_aggregates["traj_world_l2_mean"], digits=5),
                world_l2_p95=_format_float(diff_aggregates["traj_world_l2_p95"], digits=5),
                world_error_var=_format_float(diff_aggregates["traj_world_error_var_mean"], digits=5),
                step_delta=_format_float(diff_aggregates["traj_world_step_delta_l2_mean"], digits=5),
                endpoint_error=_format_float(diff_aggregates["traj_world_endpoint_l2_mean"], digits=5),
            )
        )

    lines.extend(
        [
            "",
            "## Geometry Diagnostics",
            "",
            "| Camera | Variant | 2D L2 Mean | Depth Abs Mean | Traj UVZ MAE | Common Valid Tracks | Common Valid Steps |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in summary["pairwise_comparisons"]:
        diff_aggregates = comparison["sample_diff"]["aggregates"]
        lines.append(
            "| {camera} | {variant} | {traj_2d_l2} | {traj_depth_abs} | {traj_uvz_mae} | {common_tracks} | {common_steps} |".format(
                camera=comparison["camera_name"],
                variant=comparison["variant_name"],
                traj_2d_l2=_format_float(diff_aggregates["traj_2d_l2_mean"], digits=5),
                traj_depth_abs=_format_float(diff_aggregates["traj_depth_abs_mean"], digits=5),
                traj_uvz_mae=_format_float(diff_aggregates["traj_uvz_mae_mean"], digits=5),
                common_tracks=_format_float(diff_aggregates["common_valid_track_count_mean"], digits=1),
                common_steps=_format_float(diff_aggregates["common_valid_step_count_mean"], digits=1),
            )
        )

    pareto_candidates = _find_pareto_candidates(summary)
    lines.extend(["", "## Pareto Candidates", ""])
    for camera_name in summary["camera_names"]:
        candidates = pareto_candidates.get(camera_name, [])
        if not candidates:
            lines.append(f"- `{camera_name}`: none")
            continue
        formatted = ", ".join(
            "`{variant}`(ratio={ratio}, total={total}s, jaccard={jaccard}, world_l2={world_l2})".format(
                variant=item["variant_name"],
                ratio=_format_float(item["support_grid_ratio"], digits=2),
                total=_format_float(item["total_seconds_mean"]),
                jaccard=_format_float(item["traj_valid_mask_jaccard_mean"]),
                world_l2=_format_float(item["traj_world_l2_mean"], digits=5),
            )
            for item in candidates
        )
        lines.append(f"- `{camera_name}`: {formatted}")

    if summary["verification_results"]:
        lines.extend(["", "## Verification", ""])
        for item in summary["verification_results"]:
            if item["query_frame"] is None:
                lines.append(f"- `{item['camera_name']}`: no common query frame across variants")
                continue
            lines.append(
                f"- `{item['camera_name']}`: query frame `{item['query_frame']}`"
            )
            for variant in item["variants"]:
                lines.append(
                    f"  {variant['variant_name']}: `{variant['output_dir']}`"
                )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    print("")
    print("Variant timing")
    for row in summary["variant_rows"]:
        print(
            "  {camera} / {variant}: ratio {ratio}, process {process}s, total {total}s, support mean {support_mean}".format(
                camera=row["camera_name"],
                variant=row["variant_name"],
                ratio=_format_float(row["variant_config"]["support_grid_ratio"], digits=2),
                process=_format_float(row["aggregates"]["process_seconds_mean"]),
                total=_format_float(row["aggregates"]["total_seconds_mean"]),
                support_mean=_format_float(row["aggregates"]["effective_support_query_count_mean"], digits=1),
            )
        )
    print("")
    print("Baseline diffs")
    for comparison in summary["pairwise_comparisons"]:
        diff = comparison["sample_diff"]["aggregates"]
        print(
            "  {camera} / {variant}: total {speedup}x, valid delta {valid_delta}, "
            "mask jaccard {jaccard}, world L2 mean {world_l2}".format(
                camera=comparison["camera_name"],
                variant=comparison["variant_name"],
                speedup=_format_float(comparison["total_speedup_vs_baseline"]),
                valid_delta=_format_float(diff["valid_track_count_delta_mean"]),
                jaccard=_format_float(diff["traj_valid_mask_jaccard_mean"]),
                world_l2=_format_float(diff["traj_world_l2_mean"], digits=5),
            )
        )


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    if args.episode_dir is None:
        raise ValueError("--episode-dir is required")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be >= 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.keyframes_per_sec_min <= 0 or args.keyframes_per_sec_max <= 0:
        raise ValueError("--keyframes-per-sec-min/max must both be >= 1")
    if args.keyframes_per_sec_min > args.keyframes_per_sec_max:
        raise ValueError("--keyframes-per-sec-min must be <= --keyframes-per-sec-max")

    args.episode_dir = args.episode_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    camera_names = parse_camera_names(args.camera_names)
    support_grid_ratios = parse_support_grid_ratios(args.support_grid_ratios)
    variant_names = parse_variant_names(args.variants)
    variant_specs = build_variant_specs(
        support_grid_ratios=support_grid_ratios,
        selected_variant_names=variant_names or None,
    )

    schedule_path = ensure_query_frame_schedule(
        episode_dir=args.episode_dir,
        camera_names=camera_names,
        external_geom_name=args.external_geom_name,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        keyframes_per_sec_min=args.keyframes_per_sec_min,
        keyframes_per_sec_max=args.keyframes_per_sec_max,
        keyframe_seed=args.keyframe_seed,
        fallback_episode_fps=args.fallback_episode_fps,
        output_root=args.output_root,
    )

    raw_results_dir = args.output_root / "raw_results"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[dict[str, Any]] = []
    for variant_spec in variant_specs:
        for camera_name in camera_names:
            run_label = f"{variant_spec['name']}_{camera_name}"
            result_json = raw_results_dir / f"{run_label}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--episode-dir",
                str(args.episode_dir),
                "--camera-name",
                camera_name,
                "--variant-name",
                variant_spec["name"],
                "--run-label",
                run_label,
                "--traj-filter-profile",
                args.traj_filter_profile,
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                args.device,
                "--num-iters",
                str(args.num_iters),
                "--fps",
                str(args.fps),
                "--max-num-frames",
                str(args.max_num_frames),
                "--future-len",
                str(args.future_len),
                "--grid-size",
                str(args.grid_size),
                "--filter-level",
                args.filter_level,
                "--keyframes-per-sec-min",
                str(args.keyframes_per_sec_min),
                "--keyframes-per-sec-max",
                str(args.keyframes_per_sec_max),
                "--keyframe-seed",
                str(args.keyframe_seed),
                "--fallback-episode-fps",
                str(args.fallback_episode_fps),
                "--external-geom-name",
                args.external_geom_name,
                "--external-extr-mode",
                args.external_extr_mode,
                "--benchmark-runs",
                str(args.benchmark_runs),
                "--warmup-runs",
                str(args.warmup_runs),
                "--support-grid-ratios",
                args.support_grid_ratios,
                "--output-root",
                str(args.output_root),
                "--result-json",
                str(result_json),
                "--query-frame-schedule-path",
                str(schedule_path),
            ]
            if args.keep_outputs:
                cmd.append("--keep-outputs")
            subprocess.run(cmd, check=True)
            case_results.append(json.loads(result_json.read_text(encoding="utf-8")))

    summary_by_key = {
        (case["variant_name"], case["camera_name"]): case
        for case in case_results
    }
    verification_results: list[dict[str, Any]] = []
    if args.run_visual_verification:
        verification_results = run_visual_verification(
            summary_by_key=summary_by_key,
            variant_specs=variant_specs,
            camera_names=camera_names,
            output_root=args.output_root,
        )

    summary = build_summary_payload(
        args=args,
        schedule_path=schedule_path,
        variant_specs=variant_specs,
        case_results=case_results,
        verification_results=verification_results,
    )
    summary_json_path = args.output_root / RESULT_JSON_BASENAME
    summary_md_path = args.output_root / SUMMARY_MD_BASENAME
    _atomic_write_json(summary_json_path, summary)
    write_summary_markdown(summary, summary_md_path)
    print_summary(summary)
    print("")
    print(f"JSON summary: {summary_json_path}")
    print(f"Markdown summary: {summary_md_path}")
    return summary


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    run_orchestrator(args)


if __name__ == "__main__":
    main()
