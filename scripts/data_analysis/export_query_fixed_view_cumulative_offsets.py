#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.external_wobble_diagnostics import compute_border_distance_px
from utils.query_fixed_view_depth_gate_utils import compute_query_fixed_view_depth_gate


def _parse_query_frames(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one query frame index.")
    return [int(item) for item in values]


def _load_depth_stack(depth_dir: Path) -> np.ndarray:
    depth_paths = sorted(depth_dir.glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth .npy files found under {depth_dir}")
    return np.stack([np.load(path).astype(np.float32) for path in depth_paths], axis=0).astype(np.float32)


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"finite_count": 0, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    valid = arr[finite].astype(np.float64)
    return {
        "finite_count": int(valid.size),
        "median": float(np.median(valid)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "max": float(np.max(valid)),
    }


def _masked_nanmax_rows(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"Expected values/mask to share shape, got {values.shape} vs {mask.shape}")
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    finite_rows = np.any(mask & np.isfinite(values), axis=1)
    if np.any(finite_rows):
        result[finite_rows] = np.nanmax(np.where(mask[finite_rows], values[finite_rows], np.nan), axis=1).astype(
            np.float32
        )
    return result


def _grid_shape(*, grid_size: int, trim_left: int, trim_right: int, trim_top: int, trim_bottom: int) -> tuple[int, int]:
    rows = int(grid_size) - int(trim_top) - int(trim_bottom)
    cols = int(grid_size) - int(trim_left) - int(trim_right)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid grid shape after trims: rows={rows}, cols={cols}")
    return rows, cols


def _reshape_track_metric_to_grid(
    values: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != int(grid_rows) * int(grid_cols):
        raise ValueError(f"Expected {grid_rows}x{grid_cols}={grid_rows * grid_cols} values, got {arr.size}")
    return arr.reshape(int(grid_rows), int(grid_cols)).astype(np.float32, copy=False)


def _resolve_heatmap_scale(metric_grid: np.ndarray) -> dict[str, float | None]:
    metric_grid = np.asarray(metric_grid, dtype=np.float32)
    finite = np.isfinite(metric_grid)
    if np.any(finite):
        vmin = float(np.nanmin(metric_grid))
        vmax = float(np.nanpercentile(metric_grid, 99))
        finite_max = float(np.nanmax(metric_grid))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = finite_max
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin = 0.0
        vmax = 1.0
        finite_max = None
    return {"vmin": vmin, "vmax_p99_clip": vmax, "finite_max": finite_max}


def _save_metric_heatmap(
    *,
    output_path: Path,
    metric_grid: np.ndarray,
    title: str,
    colorbar_label: str,
) -> dict[str, float | None]:
    metric_grid = np.asarray(metric_grid, dtype=np.float32)
    scale = _resolve_heatmap_scale(metric_grid)
    vmin = float(scale["vmin"])
    vmax = float(scale["vmax_p99_clip"])

    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(8.4, 6.8), dpi=160)
    image = ax.imshow(metric_grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return scale


def _save_metric_heatmap_pair(
    *,
    output_path: Path,
    left_metric_grid: np.ndarray,
    left_title: str,
    left_colorbar_label: str,
    right_metric_grid: np.ndarray,
    right_title: str,
    right_colorbar_label: str,
    suptitle: str,
) -> dict[str, dict[str, float | None]]:
    left_metric_grid = np.asarray(left_metric_grid, dtype=np.float32)
    right_metric_grid = np.asarray(right_metric_grid, dtype=np.float32)
    left_scale = _resolve_heatmap_scale(left_metric_grid)
    right_scale = _resolve_heatmap_scale(right_metric_grid)

    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad(color="#d9d9d9")

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.8), dpi=160)
    panels = [
        (
            axes[0],
            left_metric_grid,
            left_title,
            left_colorbar_label,
            left_scale,
        ),
        (
            axes[1],
            right_metric_grid,
            right_title,
            right_colorbar_label,
            right_scale,
        ),
    ]
    for ax, metric_grid, title, colorbar_label, scale in panels:
        image = ax.imshow(
            metric_grid,
            cmap=cmap,
            vmin=float(scale["vmin"]),
            vmax=float(scale["vmax_p99_clip"]),
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(title)
        ax.set_xlabel("Grid Column")
        ax.set_ylabel("Grid Row")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {"left_scale": left_scale, "right_scale": right_scale}


def _build_grid_keypoints(
    *,
    height: int,
    width: int,
    grid_size: int,
    trim_left: int = 0,
    trim_right: int = 0,
    trim_top: int = 0,
    trim_bottom: int = 0,
) -> np.ndarray:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"Expected positive height/width, got {height}x{width}")
    if int(grid_size) <= 0:
        raise ValueError(f"Expected positive grid_size, got {grid_size}")
    trims = [int(trim_left), int(trim_right), int(trim_top), int(trim_bottom)]
    if min(trims) < 0:
        raise ValueError("Expected nonnegative grid trims")
    if int(trim_left) + int(trim_right) >= int(grid_size):
        raise ValueError("Horizontal trims must leave at least one grid column")
    if int(trim_top) + int(trim_bottom) >= int(grid_size):
        raise ValueError("Vertical trims must leave at least one grid row")

    xs = np.linspace(0.0, float(width - 1), int(grid_size), dtype=np.float32)
    ys = np.linspace(0.0, float(height - 1), int(grid_size), dtype=np.float32)
    if int(trim_left) > 0:
        xs = xs[int(trim_left) :]
    if int(trim_right) > 0:
        xs = xs[: xs.shape[0] - int(trim_right)]
    if int(trim_top) > 0:
        ys = ys[int(trim_top) :]
    if int(trim_bottom) > 0:
        ys = ys[: ys.shape[0] - int(trim_bottom)]
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)


def _build_track_metrics(
    *,
    keypoints: np.ndarray,
    gate_result: dict[str, Any],
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    compare_mask = np.asarray(gate_result["compare_mask"], dtype=bool)
    query_reprojected_uvz = np.asarray(gate_result["query_reprojected_uvz"], dtype=np.float32)
    uv_delta_px = np.asarray(gate_result["uv_delta_px"], dtype=np.float32)
    depth_delta_m = np.asarray(gate_result["depth_delta_m"], dtype=np.float32)
    query_depth_values = np.asarray(gate_result["query_depth_values"], dtype=np.float32)
    query_world_valid_mask = np.asarray(gate_result["query_world_valid_mask"], dtype=bool)
    reliable_track_mask = np.asarray(gate_result["reliable_track_mask"], dtype=bool)
    compare_frame_count = np.asarray(gate_result["compare_frame_count"], dtype=np.uint16)
    depth_anomaly_hit_count = np.asarray(gate_result["depth_anomaly_hit_count"], dtype=np.uint16)
    first_anomaly_step = np.asarray(gate_result["first_anomaly_step"], dtype=np.int32)

    valid_uv_delta = np.where(compare_mask & np.isfinite(uv_delta_px), np.abs(uv_delta_px), 0.0).astype(np.float32)
    valid_depth_delta = np.where(compare_mask & np.isfinite(depth_delta_m), np.abs(depth_delta_m), 0.0).astype(
        np.float32
    )

    uv_abs_sum_px = valid_uv_delta.sum(axis=1).astype(np.float32)
    depth_abs_sum_m = valid_depth_delta.sum(axis=1).astype(np.float32)
    uv_abs_max_px = _masked_nanmax_rows(np.abs(uv_delta_px), compare_mask)
    depth_abs_max_m = _masked_nanmax_rows(np.abs(depth_delta_m), compare_mask)

    compare_frame_count_f = compare_frame_count.astype(np.float32)
    uv_abs_mean_px = np.full(keypoints.shape[0], np.nan, dtype=np.float32)
    depth_abs_mean_m = np.full(keypoints.shape[0], np.nan, dtype=np.float32)
    has_compare = compare_frame_count > 0
    uv_abs_mean_px[has_compare] = uv_abs_sum_px[has_compare] / compare_frame_count_f[has_compare]
    depth_abs_mean_m[has_compare] = depth_abs_sum_m[has_compare] / compare_frame_count_f[has_compare]

    adjacent_compare_mask = (
        compare_mask[:, 1:]
        & compare_mask[:, :-1]
        & np.isfinite(query_reprojected_uvz[:, 1:, :]).all(axis=2)
        & np.isfinite(query_reprojected_uvz[:, :-1, :]).all(axis=2)
    )
    uv_adj_step_px = np.linalg.norm(
        query_reprojected_uvz[:, 1:, :2] - query_reprojected_uvz[:, :-1, :2],
        axis=2,
    ).astype(np.float32)
    depth_adj_step_m = np.abs(query_reprojected_uvz[:, 1:, 2] - query_reprojected_uvz[:, :-1, 2]).astype(np.float32)
    valid_uv_adj_step = np.where(
        adjacent_compare_mask & np.isfinite(uv_adj_step_px),
        np.abs(uv_adj_step_px),
        0.0,
    ).astype(np.float32)
    valid_depth_adj_step = np.where(
        adjacent_compare_mask & np.isfinite(depth_adj_step_m),
        np.abs(depth_adj_step_m),
        0.0,
    ).astype(np.float32)
    adjacent_step_count = adjacent_compare_mask.sum(axis=1).astype(np.uint16)
    uv_adj_abs_sum_px = valid_uv_adj_step.sum(axis=1).astype(np.float32)
    depth_adj_abs_sum_m = valid_depth_adj_step.sum(axis=1).astype(np.float32)
    uv_adj_abs_max_px = _masked_nanmax_rows(np.abs(uv_adj_step_px), adjacent_compare_mask)
    depth_adj_abs_max_m = _masked_nanmax_rows(np.abs(depth_adj_step_m), adjacent_compare_mask)
    adjacent_step_count_f = adjacent_step_count.astype(np.float32)
    uv_adj_abs_mean_px = np.full(keypoints.shape[0], np.nan, dtype=np.float32)
    depth_adj_abs_mean_m = np.full(keypoints.shape[0], np.nan, dtype=np.float32)
    has_adjacent_step = adjacent_step_count > 0
    uv_adj_abs_mean_px[has_adjacent_step] = uv_adj_abs_sum_px[has_adjacent_step] / adjacent_step_count_f[has_adjacent_step]
    depth_adj_abs_mean_m[has_adjacent_step] = (
        depth_adj_abs_sum_m[has_adjacent_step] / adjacent_step_count_f[has_adjacent_step]
    )

    border_dist_px = compute_border_distance_px(keypoints, height=int(height), width=int(width)).astype(np.float32)

    return {
        "track_index": np.arange(keypoints.shape[0], dtype=np.int32),
        "x_px": keypoints[:, 0].astype(np.float32, copy=False),
        "y_px": keypoints[:, 1].astype(np.float32, copy=False),
        "border_dist_px": border_dist_px,
        "query_depth_m": query_depth_values.astype(np.float32, copy=False),
        "query_world_valid_mask": query_world_valid_mask.astype(bool, copy=False),
        "reliable_track_mask": reliable_track_mask.astype(bool, copy=False),
        "compare_frame_count": compare_frame_count.astype(np.uint16, copy=False),
        "adjacent_step_count": adjacent_step_count.astype(np.uint16, copy=False),
        "depth_anomaly_hit_count": depth_anomaly_hit_count.astype(np.uint16, copy=False),
        "first_anomaly_step": first_anomaly_step.astype(np.int32, copy=False),
        "uv_abs_sum_px": uv_abs_sum_px.astype(np.float32, copy=False),
        "depth_abs_sum_m": depth_abs_sum_m.astype(np.float32, copy=False),
        "uv_abs_mean_px": uv_abs_mean_px.astype(np.float32, copy=False),
        "depth_abs_mean_m": depth_abs_mean_m.astype(np.float32, copy=False),
        "uv_abs_max_px": uv_abs_max_px.astype(np.float32, copy=False),
        "depth_abs_max_m": depth_abs_max_m.astype(np.float32, copy=False),
        "uv_adj_abs_sum_px": uv_adj_abs_sum_px.astype(np.float32, copy=False),
        "depth_adj_abs_sum_m": depth_adj_abs_sum_m.astype(np.float32, copy=False),
        "uv_adj_abs_mean_px": uv_adj_abs_mean_px.astype(np.float32, copy=False),
        "depth_adj_abs_mean_m": depth_adj_abs_mean_m.astype(np.float32, copy=False),
        "uv_adj_abs_max_px": uv_adj_abs_max_px.astype(np.float32, copy=False),
        "depth_adj_abs_max_m": depth_adj_abs_max_m.astype(np.float32, copy=False),
    }


def _select_top_rows(
    *,
    track_metrics: dict[str, np.ndarray],
    sort_key: str,
    top_k: int,
) -> list[dict[str, float | int | bool | None]]:
    valid_mask = np.asarray(track_metrics["query_world_valid_mask"], dtype=bool)
    scores = np.asarray(track_metrics[sort_key], dtype=np.float32)
    finite_mask = valid_mask & np.isfinite(scores)
    if not np.any(finite_mask):
        return []
    candidate_idx = np.flatnonzero(finite_mask)
    order = np.argsort(scores[candidate_idx])[::-1]
    top_indices = candidate_idx[order[: max(int(top_k), 0)]]
    rows: list[dict[str, float | int | bool | None]] = []
    for idx in top_indices.tolist():
        rows.append(
            {
                "track_index": int(track_metrics["track_index"][idx]),
                "x_px": float(track_metrics["x_px"][idx]),
                "y_px": float(track_metrics["y_px"][idx]),
                "border_dist_px": float(track_metrics["border_dist_px"][idx]),
                "query_depth_m": float(track_metrics["query_depth_m"][idx]),
                "reliable_track": bool(track_metrics["reliable_track_mask"][idx]),
                "compare_frame_count": int(track_metrics["compare_frame_count"][idx]),
                "adjacent_step_count": int(track_metrics["adjacent_step_count"][idx]),
                "depth_anomaly_hit_count": int(track_metrics["depth_anomaly_hit_count"][idx]),
                "first_anomaly_step": int(track_metrics["first_anomaly_step"][idx]),
                "uv_abs_sum_px": float(track_metrics["uv_abs_sum_px"][idx]),
                "depth_abs_sum_m": float(track_metrics["depth_abs_sum_m"][idx]),
                "uv_abs_mean_px": float(track_metrics["uv_abs_mean_px"][idx])
                if np.isfinite(track_metrics["uv_abs_mean_px"][idx])
                else None,
                "depth_abs_mean_m": float(track_metrics["depth_abs_mean_m"][idx])
                if np.isfinite(track_metrics["depth_abs_mean_m"][idx])
                else None,
                "uv_abs_max_px": float(track_metrics["uv_abs_max_px"][idx])
                if np.isfinite(track_metrics["uv_abs_max_px"][idx])
                else None,
                "depth_abs_max_m": float(track_metrics["depth_abs_max_m"][idx])
                if np.isfinite(track_metrics["depth_abs_max_m"][idx])
                else None,
                "uv_adj_abs_sum_px": float(track_metrics["uv_adj_abs_sum_px"][idx]),
                "depth_adj_abs_sum_m": float(track_metrics["depth_adj_abs_sum_m"][idx]),
                "uv_adj_abs_mean_px": float(track_metrics["uv_adj_abs_mean_px"][idx])
                if np.isfinite(track_metrics["uv_adj_abs_mean_px"][idx])
                else None,
                "depth_adj_abs_mean_m": float(track_metrics["depth_adj_abs_mean_m"][idx])
                if np.isfinite(track_metrics["depth_adj_abs_mean_m"][idx])
                else None,
                "uv_adj_abs_max_px": float(track_metrics["uv_adj_abs_max_px"][idx])
                if np.isfinite(track_metrics["uv_adj_abs_max_px"][idx])
                else None,
                "depth_adj_abs_max_m": float(track_metrics["depth_adj_abs_max_m"][idx])
                if np.isfinite(track_metrics["depth_adj_abs_max_m"][idx])
                else None,
            }
        )
    return rows


def _write_track_csv(csv_path: Path, track_metrics: dict[str, np.ndarray]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_index",
        "x_px",
        "y_px",
        "border_dist_px",
        "query_depth_m",
        "query_world_valid_mask",
        "reliable_track_mask",
        "compare_frame_count",
        "adjacent_step_count",
        "depth_anomaly_hit_count",
        "first_anomaly_step",
        "uv_abs_sum_px",
        "depth_abs_sum_m",
        "uv_abs_mean_px",
        "depth_abs_mean_m",
        "uv_abs_max_px",
        "depth_abs_max_m",
        "uv_adj_abs_sum_px",
        "depth_adj_abs_sum_m",
        "uv_adj_abs_mean_px",
        "depth_adj_abs_mean_m",
        "uv_adj_abs_max_px",
        "depth_adj_abs_max_m",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        track_count = int(track_metrics["track_index"].shape[0])
        for idx in range(track_count):
            writer.writerow(
                {
                    "track_index": int(track_metrics["track_index"][idx]),
                    "x_px": float(track_metrics["x_px"][idx]),
                    "y_px": float(track_metrics["y_px"][idx]),
                    "border_dist_px": float(track_metrics["border_dist_px"][idx]),
                    "query_depth_m": float(track_metrics["query_depth_m"][idx])
                    if np.isfinite(track_metrics["query_depth_m"][idx])
                    else "",
                    "query_world_valid_mask": bool(track_metrics["query_world_valid_mask"][idx]),
                    "reliable_track_mask": bool(track_metrics["reliable_track_mask"][idx]),
                    "compare_frame_count": int(track_metrics["compare_frame_count"][idx]),
                    "adjacent_step_count": int(track_metrics["adjacent_step_count"][idx]),
                    "depth_anomaly_hit_count": int(track_metrics["depth_anomaly_hit_count"][idx]),
                    "first_anomaly_step": int(track_metrics["first_anomaly_step"][idx]),
                    "uv_abs_sum_px": float(track_metrics["uv_abs_sum_px"][idx])
                    if np.isfinite(track_metrics["uv_abs_sum_px"][idx])
                    else "",
                    "depth_abs_sum_m": float(track_metrics["depth_abs_sum_m"][idx])
                    if np.isfinite(track_metrics["depth_abs_sum_m"][idx])
                    else "",
                    "uv_abs_mean_px": float(track_metrics["uv_abs_mean_px"][idx])
                    if np.isfinite(track_metrics["uv_abs_mean_px"][idx])
                    else "",
                    "depth_abs_mean_m": float(track_metrics["depth_abs_mean_m"][idx])
                    if np.isfinite(track_metrics["depth_abs_mean_m"][idx])
                    else "",
                    "uv_abs_max_px": float(track_metrics["uv_abs_max_px"][idx])
                    if np.isfinite(track_metrics["uv_abs_max_px"][idx])
                    else "",
                    "depth_abs_max_m": float(track_metrics["depth_abs_max_m"][idx])
                    if np.isfinite(track_metrics["depth_abs_max_m"][idx])
                    else "",
                    "uv_adj_abs_sum_px": float(track_metrics["uv_adj_abs_sum_px"][idx])
                    if np.isfinite(track_metrics["uv_adj_abs_sum_px"][idx])
                    else "",
                    "depth_adj_abs_sum_m": float(track_metrics["depth_adj_abs_sum_m"][idx])
                    if np.isfinite(track_metrics["depth_adj_abs_sum_m"][idx])
                    else "",
                    "uv_adj_abs_mean_px": float(track_metrics["uv_adj_abs_mean_px"][idx])
                    if np.isfinite(track_metrics["uv_adj_abs_mean_px"][idx])
                    else "",
                    "depth_adj_abs_mean_m": float(track_metrics["depth_adj_abs_mean_m"][idx])
                    if np.isfinite(track_metrics["depth_adj_abs_mean_m"][idx])
                    else "",
                    "uv_adj_abs_max_px": float(track_metrics["uv_adj_abs_max_px"][idx])
                    if np.isfinite(track_metrics["uv_adj_abs_max_px"][idx])
                    else "",
                    "depth_adj_abs_max_m": float(track_metrics["depth_adj_abs_max_m"][idx])
                    if np.isfinite(track_metrics["depth_adj_abs_max_m"][idx])
                    else "",
                }
            )


def _default_case_output_dir(case_dir: Path, *, grid_size: int, query_frames: list[int]) -> Path:
    query_suffix = "_".join(f"q{frame}" for frame in query_frames)
    return case_dir / "_analysis_query_fixed_view_cumulative_offsets" / f"grid{int(grid_size)}_{query_suffix}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-keypoint fixed-view cumulative UV/depth offsets over time for one or more xperience cases."
        )
    )
    parser.add_argument(
        "--case_dir",
        type=Path,
        action="append",
        required=True,
        help="Case directory. Repeat the flag to analyze multiple cases.",
    )
    parser.add_argument("--camera_name", type=str, default="stereo_left")
    parser.add_argument("--query_frames", type=str, default="0,4")
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--trim_left", type=int, default=0)
    parser.add_argument("--trim_right", type=int, default=0)
    parser.add_argument("--trim_top", type=int, default=0)
    parser.add_argument("--trim_bottom", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional aggregate output root. Per-case artifacts are still written under each case unless overridden.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    query_frames = _parse_query_frames(args.query_frames)
    case_payloads: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for raw_case_dir in args.case_dir:
        case_dir = raw_case_dir.resolve()
        depth_dir = case_dir / "depth" / str(args.camera_name)
        geom_npz = case_dir / "geom" / f"geom_{args.camera_name}_official_w2c.npz"
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
        if not geom_npz.is_file():
            raise FileNotFoundError(f"Missing geometry file: {geom_npz}")

        depth_frames = _load_depth_stack(depth_dir)
        geom = np.load(geom_npz)
        try:
            intrinsics = np.asarray(geom["intrinsics"], dtype=np.float32)
            extrinsics = np.asarray(geom["extrinsics"], dtype=np.float32)
        finally:
            geom.close()

        frame_count, height, width = depth_frames.shape
        keypoints = _build_grid_keypoints(
            height=height,
            width=width,
            grid_size=int(args.grid_size),
            trim_left=int(args.trim_left),
            trim_right=int(args.trim_right),
            trim_top=int(args.trim_top),
            trim_bottom=int(args.trim_bottom),
        )

        case_output_dir = _default_case_output_dir(case_dir, grid_size=int(args.grid_size), query_frames=query_frames)
        case_output_dir.mkdir(parents=True, exist_ok=True)
        query_payloads: list[dict[str, Any]] = []

        for query_frame in query_frames:
            gate_result = compute_query_fixed_view_depth_gate(
                depth_frames,
                intrinsics,
                extrinsics,
                keypoints=keypoints,
                query_frame=int(query_frame),
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
            )
            track_metrics = _build_track_metrics(
                keypoints=keypoints,
                gate_result=gate_result,
                height=height,
                width=width,
            )
            valid_mask = np.asarray(track_metrics["query_world_valid_mask"], dtype=bool)
            reliable_mask = np.asarray(track_metrics["reliable_track_mask"], dtype=bool)
            query_payload = {
                "query_frame": int(query_frame),
                "frame_count": int(frame_count),
                "grid_size": int(args.grid_size),
                "grid_track_count": int(keypoints.shape[0]),
                "query_world_valid_count": int(np.count_nonzero(valid_mask)),
                "reliable_track_count": int(np.count_nonzero(reliable_mask)),
                "removed_track_count": int(np.count_nonzero(valid_mask & (~reliable_mask))),
                "compare_frame_count_summary": _finite_summary(track_metrics["compare_frame_count"][valid_mask]),
                "adjacent_step_count_summary": _finite_summary(track_metrics["adjacent_step_count"][valid_mask]),
                "query_depth_m_summary": _finite_summary(track_metrics["query_depth_m"][valid_mask]),
                "border_dist_px_summary": _finite_summary(track_metrics["border_dist_px"][valid_mask]),
                "uv_abs_sum_px_summary": _finite_summary(track_metrics["uv_abs_sum_px"][valid_mask]),
                "depth_abs_sum_m_summary": _finite_summary(track_metrics["depth_abs_sum_m"][valid_mask]),
                "uv_abs_mean_px_summary": _finite_summary(track_metrics["uv_abs_mean_px"][valid_mask]),
                "depth_abs_mean_m_summary": _finite_summary(track_metrics["depth_abs_mean_m"][valid_mask]),
                "uv_abs_max_px_summary": _finite_summary(track_metrics["uv_abs_max_px"][valid_mask]),
                "depth_abs_max_m_summary": _finite_summary(track_metrics["depth_abs_max_m"][valid_mask]),
                "uv_adj_abs_sum_px_summary": _finite_summary(track_metrics["uv_adj_abs_sum_px"][valid_mask]),
                "depth_adj_abs_sum_m_summary": _finite_summary(track_metrics["depth_adj_abs_sum_m"][valid_mask]),
                "uv_adj_abs_mean_px_summary": _finite_summary(track_metrics["uv_adj_abs_mean_px"][valid_mask]),
                "depth_adj_abs_mean_m_summary": _finite_summary(track_metrics["depth_adj_abs_mean_m"][valid_mask]),
                "uv_adj_abs_max_px_summary": _finite_summary(track_metrics["uv_adj_abs_max_px"][valid_mask]),
                "depth_adj_abs_max_m_summary": _finite_summary(track_metrics["depth_adj_abs_max_m"][valid_mask]),
                "top_uv_abs_sum_tracks": _select_top_rows(
                    track_metrics=track_metrics,
                    sort_key="uv_abs_sum_px",
                    top_k=int(args.top_k),
                ),
                "top_depth_abs_sum_tracks": _select_top_rows(
                    track_metrics=track_metrics,
                    sort_key="depth_abs_sum_m",
                    top_k=int(args.top_k),
                ),
                "top_uv_adj_abs_sum_tracks": _select_top_rows(
                    track_metrics=track_metrics,
                    sort_key="uv_adj_abs_sum_px",
                    top_k=int(args.top_k),
                ),
                "top_depth_adj_abs_sum_tracks": _select_top_rows(
                    track_metrics=track_metrics,
                    sort_key="depth_adj_abs_sum_m",
                    top_k=int(args.top_k),
                ),
            }
            query_payloads.append(query_payload)

            csv_path = case_output_dir / f"q{int(query_frame):05d}_track_metrics.csv"
            _write_track_csv(csv_path, track_metrics)
            grid_rows, grid_cols = _grid_shape(
                grid_size=int(args.grid_size),
                trim_left=int(args.trim_left),
                trim_right=int(args.trim_right),
                trim_top=int(args.trim_top),
                trim_bottom=int(args.trim_bottom),
            )
            valid_track_grid = _reshape_track_metric_to_grid(
                np.where(
                    np.asarray(track_metrics["query_world_valid_mask"], dtype=bool),
                    np.asarray(track_metrics["uv_adj_abs_sum_px"], dtype=np.float32),
                    np.nan,
                ),
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )
            uv_heatmap_path = case_output_dir / f"q{int(query_frame):05d}_uv_adj_abs_sum_heatmap.png"
            uv_heatmap_scale = _save_metric_heatmap(
                output_path=uv_heatmap_path,
                metric_grid=valid_track_grid,
                title=f"{case_dir.name} q={int(query_frame)} uv adjacent abs sum",
                colorbar_label="uv_adj_abs_sum_px",
            )
            valid_depth_grid = _reshape_track_metric_to_grid(
                np.where(
                    np.asarray(track_metrics["query_world_valid_mask"], dtype=bool),
                    np.asarray(track_metrics["depth_adj_abs_sum_m"], dtype=np.float32),
                    np.nan,
                ),
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )
            depth_heatmap_path = case_output_dir / f"q{int(query_frame):05d}_depth_adj_abs_sum_heatmap.png"
            depth_heatmap_scale = _save_metric_heatmap(
                output_path=depth_heatmap_path,
                metric_grid=valid_depth_grid,
                title=f"{case_dir.name} q={int(query_frame)} depth adjacent abs sum",
                colorbar_label="depth_adj_abs_sum_m",
            )
            compare_heatmap_path = case_output_dir / f"q{int(query_frame):05d}_uv_depth_adj_abs_sum_compare.png"
            _save_metric_heatmap_pair(
                output_path=compare_heatmap_path,
                left_metric_grid=valid_track_grid,
                left_title="UV adjacent abs sum",
                left_colorbar_label="uv_adj_abs_sum_px",
                right_metric_grid=valid_depth_grid,
                right_title="Depth adjacent abs sum",
                right_colorbar_label="depth_adj_abs_sum_m",
                suptitle=f"{case_dir.name} q={int(query_frame)} adjacent abs sum comparison",
            )
            query_payload["heatmaps"] = {
                "uv_adj_abs_sum_heatmap_png": str(uv_heatmap_path),
                "depth_adj_abs_sum_heatmap_png": str(depth_heatmap_path),
                "uv_depth_adj_abs_sum_compare_png": str(compare_heatmap_path),
                "uv_adj_abs_sum_scale": uv_heatmap_scale,
                "depth_adj_abs_sum_scale": depth_heatmap_scale,
            }

            aggregate_rows.append(
                {
                    "case_dir": str(case_dir),
                    "case_name": case_dir.name,
                    "query_frame": int(query_frame),
                    "query_world_valid_count": query_payload["query_world_valid_count"],
                    "uv_abs_sum_px_median": query_payload["uv_abs_sum_px_summary"]["median"],
                    "uv_abs_sum_px_p95": query_payload["uv_abs_sum_px_summary"]["p95"],
                    "uv_abs_sum_px_p99": query_payload["uv_abs_sum_px_summary"]["p99"],
                    "uv_abs_sum_px_max": query_payload["uv_abs_sum_px_summary"]["max"],
                    "depth_abs_sum_m_median": query_payload["depth_abs_sum_m_summary"]["median"],
                    "depth_abs_sum_m_p95": query_payload["depth_abs_sum_m_summary"]["p95"],
                    "depth_abs_sum_m_p99": query_payload["depth_abs_sum_m_summary"]["p99"],
                    "depth_abs_sum_m_max": query_payload["depth_abs_sum_m_summary"]["max"],
                    "uv_abs_mean_px_p95": query_payload["uv_abs_mean_px_summary"]["p95"],
                    "depth_abs_mean_m_p95": query_payload["depth_abs_mean_m_summary"]["p95"],
                    "uv_adj_abs_sum_px_p95": query_payload["uv_adj_abs_sum_px_summary"]["p95"],
                    "uv_adj_abs_sum_px_p99": query_payload["uv_adj_abs_sum_px_summary"]["p99"],
                    "uv_adj_abs_sum_px_max": query_payload["uv_adj_abs_sum_px_summary"]["max"],
                    "depth_adj_abs_sum_m_p95": query_payload["depth_adj_abs_sum_m_summary"]["p95"],
                    "depth_adj_abs_sum_m_p99": query_payload["depth_adj_abs_sum_m_summary"]["p99"],
                    "depth_adj_abs_sum_m_max": query_payload["depth_adj_abs_sum_m_summary"]["max"],
                    "uv_adj_abs_mean_px_p95": query_payload["uv_adj_abs_mean_px_summary"]["p95"],
                    "depth_adj_abs_mean_m_p95": query_payload["depth_adj_abs_mean_m_summary"]["p95"],
                }
            )

        case_payload = {
            "case_dir": str(case_dir),
            "case_name": case_dir.name,
            "camera_name": str(args.camera_name),
            "depth_dir": str(depth_dir),
            "geom_npz": str(geom_npz),
            "depth_shape": [int(frame_count), int(height), int(width)],
            "grid_size": int(args.grid_size),
            "trim_left": int(args.trim_left),
            "trim_right": int(args.trim_right),
            "trim_top": int(args.trim_top),
            "trim_bottom": int(args.trim_bottom),
            "query_frames": [int(frame) for frame in query_frames],
            "query_reports": query_payloads,
            "note": (
                "uv/depth cumulative offsets are summed over frames where the fixed-view comparison is valid; "
                "see compare_frame_count for first-frame-referenced coverage and adjacent_step_count for consecutive-frame coverage."
            ),
        }
        case_payloads.append(case_payload)
        (case_output_dir / "summary.json").write_text(
            json.dumps(case_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    aggregate_payload = {
        "camera_name": str(args.camera_name),
        "query_frames": [int(frame) for frame in query_frames],
        "grid_size": int(args.grid_size),
        "trim_left": int(args.trim_left),
        "trim_right": int(args.trim_right),
        "trim_top": int(args.trim_top),
        "trim_bottom": int(args.trim_bottom),
        "cases": case_payloads,
        "aggregate_rows": aggregate_rows,
    }
    stdout_payload = {
        "camera_name": str(args.camera_name),
        "query_frames": [int(frame) for frame in query_frames],
        "grid_size": int(args.grid_size),
        "case_count": len(case_payloads),
        "aggregate_rows": aggregate_rows,
        "output_root": str(args.output_root.resolve()) if args.output_root is not None else None,
    }
    print(json.dumps(stdout_payload, indent=2, ensure_ascii=False))

    if args.output_root is not None:
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "aggregate_summary.json").write_text(
            json.dumps(aggregate_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
