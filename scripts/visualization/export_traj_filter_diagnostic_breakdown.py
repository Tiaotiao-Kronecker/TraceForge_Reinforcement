#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.query_sampling_utils import (
    QUERY_RISK_DEPTH_EDGE,
    QUERY_RISK_IMAGE_BORDER,
    QUERY_RISK_LOW_TEXTURE,
    QUERY_RISK_SPECULAR,
    QUERY_SOURCE_CONTEXT,
    QUERY_SOURCE_FALLBACK_GRID,
    QUERY_SOURCE_HAND,
    QUERY_SOURCE_INTERACTION_OBJECT,
    QUERY_SOURCE_MANIPULATOR,
)
from utils.traceforge_artifact_utils import SceneReader, normalize_sample_data
from utils.traj_filter_utils import (
    MASK_REASON_BASE_GEOMETRY_FAIL,
    MASK_REASON_MANIPULATOR_CLUSTER_FAIL,
    MASK_REASON_MANIPULATOR_DEPTH_FAIL,
    MASK_REASON_MANIPULATOR_MOTION_FAIL,
    MASK_REASON_QUERY_DEPTH_EDGE_FAIL,
    MASK_REASON_QUERY_DEPTH_FAIL,
    MASK_REASON_STABLE_TEMPORAL_FAIL,
    MASK_REASON_TEMPORAL_CONSISTENCY_FAIL,
    compute_traj_base_geometry,
)


STAGE_SPECS: list[tuple[str, str]] = [
    ("traj_query_fixed_view_depth_consistency_mask", "fixed_view_depth"),
    ("traj_base_mask", "base"),
    ("traj_query_depth_quality_mask", "query_depth_quality"),
    ("traj_query_depth_keep_mask", "query_depth_keep"),
    ("traj_supervision_support_mask", "supervision_support"),
    ("traj_wrist_seed_mask", "wrist_seed"),
    ("traj_near_depth_mask", "near_depth"),
    ("traj_motion_mask", "motion"),
    ("traj_cluster_mask", "cluster"),
    ("traj_pre_top95_mask", "pre_top95"),
    ("traj_stereo_consistency_mask", "stereo_consistency"),
    ("traj_pick_place_object_mask", "pick_place_object"),
    ("traj_valid_mask", "final"),
]

STAGE_REACH_SPECS: list[tuple[str, str, str]] = [
    ("traj_query_fixed_view_depth_consistency_mask", "fixed_view_depth", "#1f77b4"),
    ("traj_base_mask", "base", "#95a5a6"),
    ("traj_query_depth_keep_mask", "query_depth_keep", "#3498db"),
    ("traj_supervision_support_mask", "supervision_support", "#9b59b6"),
    ("traj_wrist_seed_mask", "wrist_seed", "#e67e22"),
    ("traj_near_depth_mask", "near_depth", "#d35400"),
    ("traj_motion_mask", "motion", "#f1c40f"),
    ("traj_cluster_mask", "cluster", "#16a085"),
    ("traj_pre_top95_mask", "pre_top95", "#27ae60"),
    ("traj_valid_mask", "final", "#00c853"),
]

REASON_BIT_SPECS: list[tuple[int, str]] = [
    (int(MASK_REASON_BASE_GEOMETRY_FAIL), "base_fail"),
    (int(MASK_REASON_QUERY_DEPTH_FAIL), "query_depth_fail"),
    (int(MASK_REASON_TEMPORAL_CONSISTENCY_FAIL), "temporal_fail"),
    (int(MASK_REASON_STABLE_TEMPORAL_FAIL), "stable_temporal_fail"),
    (int(MASK_REASON_MANIPULATOR_DEPTH_FAIL), "manip_depth_fail"),
    (int(MASK_REASON_MANIPULATOR_MOTION_FAIL), "manip_motion_fail"),
    (int(MASK_REASON_MANIPULATOR_CLUSTER_FAIL), "manip_cluster_fail"),
    (int(MASK_REASON_QUERY_DEPTH_EDGE_FAIL), "query_edge_fail"),
]

SOURCE_BIT_SPECS: list[tuple[int, str]] = [
    (int(QUERY_SOURCE_HAND), "hand"),
    (int(QUERY_SOURCE_MANIPULATOR), "manipulator"),
    (int(QUERY_SOURCE_INTERACTION_OBJECT), "interaction_object"),
    (int(QUERY_SOURCE_CONTEXT), "context"),
    (int(QUERY_SOURCE_FALLBACK_GRID), "fallback_grid"),
]

RISK_BIT_SPECS: list[tuple[int, str]] = [
    (int(QUERY_RISK_LOW_TEXTURE), "low_texture"),
    (int(QUERY_RISK_SPECULAR), "specular"),
    (int(QUERY_RISK_DEPTH_EDGE), "depth_edge"),
    (int(QUERY_RISK_IMAGE_BORDER), "image_border"),
]


def parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one query frame index.")
    return [int(item) for item in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-query trajectory-filter diagnostic breakdown figures and stats."
    )
    parser.add_argument(
        "--trajectory_output_dir",
        type=Path,
        required=True,
        help="Path to one camera output directory, e.g. <episode>/trajectory_dense_xxx/stereo_left.",
    )
    parser.add_argument(
        "--query_frames",
        type=str,
        required=True,
        help="Comma-separated query frame indices to export.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--point_size", type=float, default=10.0)
    parser.add_argument("--alpha_kept", type=float, default=0.9)
    parser.add_argument("--alpha_dropped", type=float, default=0.18)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--summary_name", type=str, default="summary.json")
    return parser


def _load_sample_path(trajectory_output_dir: Path, video_name: str, query_frame: int) -> Path:
    sample_path = trajectory_output_dir / "samples" / f"{video_name}_{int(query_frame)}.npz"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing sample: {sample_path}")
    return sample_path


def _mask_or_none(
    sample: dict[str, object],
    field_name: str,
    count: int,
    *,
    present_fields: set[str] | None = None,
) -> np.ndarray | None:
    if present_fields is not None and field_name not in present_fields:
        return None
    value = sample.get(field_name)
    if value is None:
        return None
    mask = np.asarray(value).astype(bool, copy=False).reshape(-1)
    if mask.shape != (count,):
        return None
    return mask


def _count_true(mask: np.ndarray | None) -> int | None:
    if mask is None:
        return None
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _finite_float_summary(values: np.ndarray | None) -> dict[str, float | int]:
    if values is None:
        return {"finite_count": 0}
    arr = np.asarray(values)
    if arr.size == 0:
        return {"finite_count": 0}
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"finite_count": 0}
    valid = arr[finite].astype(np.float64)
    return {
        "finite_count": int(valid.size),
        "min": float(np.min(valid)),
        "median": float(np.median(valid)),
        "mean": float(np.mean(valid)),
        "p90": float(np.percentile(valid, 90)),
        "max": float(np.max(valid)),
    }


def _exact_hist(values: np.ndarray | None) -> dict[str, int]:
    if values is None:
        return {}
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {}
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, dtype=bool)
    arr = arr[finite]
    if arr.size == 0:
        return {}
    uniq, counts = np.unique(arr.astype(np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(uniq.tolist(), counts.tolist(), strict=False)}


def _bit_count_hist(values: np.ndarray | None, specs: list[tuple[int, str]]) -> dict[str, int]:
    if values is None:
        return {}
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {}
    arr = arr.astype(np.int64, copy=False)
    return {label: int(np.count_nonzero((arr & bit) != 0)) for bit, label in specs}


def _bool_counts(mapping: dict[str, np.ndarray]) -> dict[str, int]:
    return {key: int(np.count_nonzero(np.asarray(value, dtype=bool))) for key, value in mapping.items()}


def _compute_base_geometry_debug(
    *,
    traj_uvz: np.ndarray,
    visibility: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> dict[str, object]:
    kwargs = {
        "traj": traj_uvz,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "min_valid_frames": 3,
        "min_depth": 0.01,
        "max_depth": 10.0,
        "boundary_margin": 50,
        "visibility_threshold": 0.5,
        "check_depth_smoothness": True,
        "depth_change_threshold": 0.5,
    }
    no_vis = compute_traj_base_geometry(visibs=None, **kwargs)
    result: dict[str, object] = {
        "without_visibility": _bool_counts(
            {
                "traj_valid_mask": no_vis["traj_valid_mask"],
                "valid_count_mask": no_vis["valid_count_mask"],
                "depth_range_mask": no_vis["depth_range_mask"],
                "boundary_mask": no_vis["boundary_mask"],
                "depth_smooth_mask": no_vis["depth_smooth_mask"],
            }
        )
    }
    if visibility is None:
        return result

    with_vis = compute_traj_base_geometry(visibs=visibility, **kwargs)
    valid = np.isfinite(traj_uvz).all(axis=-1)
    vis = np.asarray(visibility, dtype=np.float32)
    vis_bool = vis >= 0.5
    vis_count = (vis_bool & valid).sum(axis=1).astype(np.float32)
    valid_count = valid.sum(axis=1).astype(np.float32)
    vis_ratio = np.divide(
        vis_count,
        np.maximum(valid_count, 1.0),
        out=np.zeros_like(vis_count),
        where=valid_count > 0,
    )
    vis_frame_hist = _integer_hist(vis_bool.sum(axis=1))
    result["with_visibility"] = _bool_counts(
        {
            "traj_valid_mask": with_vis["traj_valid_mask"],
            "valid_count_mask": with_vis["valid_count_mask"],
            "depth_range_mask": with_vis["depth_range_mask"],
            "boundary_mask": with_vis["boundary_mask"],
            "visibility_mask": with_vis["visibility_mask"],
            "depth_smooth_mask": with_vis["depth_smooth_mask"],
        }
    )
    result["visibility_ratio_summary"] = _finite_float_summary(vis_ratio)
    result["visibility_true_frame_hist"] = {str(k): v for k, v in vis_frame_hist.items()}
    result["raw_visibility_summary"] = _finite_float_summary(vis)
    return result


def _integer_hist(values: np.ndarray | None) -> dict[int, int]:
    if values is None:
        return {}
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {}
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, dtype=bool)
    arr = arr[finite]
    if arr.size == 0:
        return {}
    uniq, counts = np.unique(arr.astype(np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(uniq.tolist(), counts.tolist(), strict=False)}


def _compute_reach_stage(
    sample: dict[str, object],
    track_count: int,
    *,
    present_fields: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    stage_index = np.full(track_count, -1, dtype=np.int16)
    stage_hist: dict[str, int] = {}
    for idx, (field_name, label, _color) in enumerate(STAGE_REACH_SPECS):
        mask = _mask_or_none(sample, field_name, track_count, present_fields=present_fields)
        if mask is None:
            continue
        stage_index[mask] = np.int16(idx)
        stage_hist[label] = int(np.count_nonzero(mask))
    rejected_count = int(np.count_nonzero(stage_index < 0))
    if rejected_count > 0:
        stage_hist["rejected_before_base"] = rejected_count
    return stage_index, stage_hist


def _plot_mask_panel(
    ax,
    *,
    rgb: np.ndarray,
    keypoints: np.ndarray,
    mask: np.ndarray | None,
    title: str,
    point_size: float,
    alpha_kept: float,
    alpha_dropped: float,
) -> None:
    ax.imshow(rgb)
    ax.set_axis_off()
    if mask is None:
        ax.set_title(f"{title}\nmissing", fontsize=10)
        return
    dropped = ~mask
    if np.any(dropped):
        ax.scatter(
            keypoints[dropped, 0],
            keypoints[dropped, 1],
            s=point_size,
            c="#ffffff",
            alpha=alpha_dropped,
            linewidths=0,
        )
    if np.any(mask):
        ax.scatter(
            keypoints[mask, 0],
            keypoints[mask, 1],
            s=point_size,
            c="#00c853",
            alpha=alpha_kept,
            linewidths=0,
        )
    ax.set_title(f"{title}\nkeep={int(np.count_nonzero(mask))}", fontsize=10)


def _render_stage_panel_figure(
    *,
    output_path: Path,
    rgb: np.ndarray,
    sample: dict[str, object],
    keypoints: np.ndarray,
    query_frame: int,
    point_size: float,
    alpha_kept: float,
    alpha_dropped: float,
    dpi: int,
    present_fields: set[str] | None,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for ax, (field_name, label) in zip(axes.flat, STAGE_SPECS, strict=False):
        _plot_mask_panel(
            ax,
            rgb=rgb,
            keypoints=keypoints,
            mask=_mask_or_none(sample, field_name, keypoints.shape[0], present_fields=present_fields),
            title=label,
            point_size=point_size,
            alpha_kept=alpha_kept,
            alpha_dropped=alpha_dropped,
        )
    fig.suptitle(f"q={query_frame:05d} stage masks", fontsize=14)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_stage_reach_overlay(
    ax,
    *,
    rgb: np.ndarray,
    keypoints: np.ndarray,
    stage_index: np.ndarray,
) -> dict[str, int]:
    ax.imshow(rgb)
    ax.set_axis_off()
    category_counts: dict[str, int] = {}
    rejected_mask = stage_index < 0
    if np.any(rejected_mask):
        ax.scatter(
            keypoints[rejected_mask, 0],
            keypoints[rejected_mask, 1],
            s=10.0,
            c="#ff5252",
            alpha=0.5,
            linewidths=0,
            label="rejected_before_base",
        )
        category_counts["rejected_before_base"] = int(np.count_nonzero(rejected_mask))
    for idx, (_field_name, label, color) in enumerate(STAGE_REACH_SPECS):
        mask = stage_index == idx
        if not np.any(mask):
            continue
        ax.scatter(
            keypoints[mask, 0],
            keypoints[mask, 1],
            s=10.0,
            c=color,
            alpha=0.8,
            linewidths=0,
            label=label,
        )
        category_counts[label] = int(np.count_nonzero(mask))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_title("farthest stage reached", fontsize=11)
    return category_counts


def _plot_stage_counts(ax, stage_counts: dict[str, int | None]) -> None:
    labels = [label for _field, label in STAGE_SPECS]
    values = [0 if stage_counts.get(label) is None else int(stage_counts[label]) for label in labels]
    y = np.arange(len(labels))
    ax.barh(y, values, color="#1976d2")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("track count")
    ax.set_title("stage counts", fontsize=11)
    for yy, value in zip(y.tolist(), values, strict=False):
        ax.text(value + 1, yy, str(value), va="center", fontsize=8)


def _plot_grouped_integer_hist(
    ax,
    *,
    series: list[tuple[str, np.ndarray | None, str]],
    title: str,
) -> None:
    merged_keys: set[int] = set()
    histograms: list[tuple[str, dict[int, int], str]] = []
    for label, values, color in series:
        hist = _integer_hist(values)
        histograms.append((label, hist, color))
        merged_keys.update(hist.keys())
    xs = sorted(merged_keys)
    if not xs:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return
    base_x = np.arange(len(xs), dtype=np.float32)
    width = 0.8 / max(len(histograms), 1)
    offsets = np.linspace(-0.4 + width / 2.0, 0.4 - width / 2.0, num=len(histograms), dtype=np.float32)
    for offset, (label, hist, color) in zip(offsets.tolist(), histograms, strict=False):
        heights = [hist.get(x, 0) for x in xs]
        ax.bar(base_x + offset, heights, width=width, label=label, color=color, alpha=0.8)
    ax.set_xticks(base_x, [str(x) for x in xs])
    ax.set_xlabel("value")
    ax.set_ylabel("track count")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)


def _plot_bit_groups(
    ax,
    *,
    reason_counts: dict[str, int],
    source_counts: dict[str, int],
    risk_counts: dict[str, int],
) -> None:
    sections = [
        ("reason", reason_counts, "#d32f2f"),
        ("source", source_counts, "#1976d2"),
        ("risk", risk_counts, "#f57c00"),
    ]
    labels: list[str] = []
    values: list[int] = []
    colors: list[str] = []
    separators: list[int] = []
    cursor = 0
    for prefix, counts, color in sections:
        for key, value in counts.items():
            labels.append(f"{prefix}:{key}")
            values.append(int(value))
            colors.append(color)
            cursor += 1
        separators.append(cursor - 0.5)
    if not labels:
        ax.text(0.5, 0.5, "no bit stats", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("reason/source/risk bit counts", fontsize=11)
        return
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, alpha=0.85)
    for sep in separators[:-1]:
        ax.axvline(sep, color="#cccccc", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("track count")
    ax.set_title("reason/source/risk bit counts", fontsize=11)


def _render_dashboard_figure(
    *,
    output_path: Path,
    rgb: np.ndarray,
    sample: dict[str, object],
    keypoints: np.ndarray,
    query_frame: int,
    stage_counts: dict[str, int | None],
    stage_reach_hist: dict[str, int],
    reason_counts: dict[str, int],
    source_counts: dict[str, int],
    risk_counts: dict[str, int],
    dpi: int,
    present_fields: set[str] | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    stage_index, _ = _compute_reach_stage(sample, keypoints.shape[0], present_fields=present_fields)
    _plot_stage_reach_overlay(axes[0, 0], rgb=rgb, keypoints=keypoints, stage_index=stage_index)
    _plot_stage_counts(axes[0, 1], stage_counts)
    _plot_grouped_integer_hist(
        axes[1, 0],
        series=[
            ("compare", sample.get("traj_compare_frame_count"), "#6a1b9a"),
            ("prefix", sample.get("traj_supervision_prefix_len"), "#00897b"),
            ("stereo_compare", sample.get("traj_stereo_compare_frame_count"), "#ef6c00"),
        ],
        title="temporal support histograms",
    )
    _plot_bit_groups(
        axes[1, 1],
        reason_counts=reason_counts,
        source_counts=source_counts,
        risk_counts=risk_counts,
    )
    valid_count = _count_true(_mask_or_none(sample, "traj_valid_mask", keypoints.shape[0])) or 0
    fig.suptitle(
        f"q={query_frame:05d} diagnostic dashboard | tracks={keypoints.shape[0]} | valid={valid_count} | "
        f"reach={stage_reach_hist}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _query_summary(
    *,
    trajectory_output_dir: Path,
    sample_path: Path,
    sample: dict[str, object],
    stage_counts: dict[str, int | None],
    stage_reach_hist: dict[str, int],
    reason_counts: dict[str, int],
    source_counts: dict[str, int],
    risk_counts: dict[str, int],
    figures: dict[str, str],
    missing_stage_fields: list[str],
    base_geometry_debug: dict[str, object],
) -> dict[str, object]:
    return {
        "trajectory_output_dir": str(trajectory_output_dir),
        "sample_path": str(sample_path),
        "query_frame_index": int(sample["query_frame_index"]),
        "dense_query_count": int(sample["dense_query_count"]),
        "tracked_query_count": int(sample["tracked_query_count"]),
        "valid_track_count": int(np.count_nonzero(np.asarray(sample["traj_valid_mask"], dtype=bool))),
        "missing_stage_fields": missing_stage_fields,
        "stage_counts": stage_counts,
        "stage_reach_hist": stage_reach_hist,
        "reason_value_hist": _exact_hist(sample.get("traj_mask_reason_bits")),
        "reason_bit_counts": reason_counts,
        "query_source_value_hist": _exact_hist(sample.get("traj_query_source_bits")),
        "query_source_bit_counts": source_counts,
        "query_risk_value_hist": _exact_hist(sample.get("traj_query_risk_bits")),
        "query_risk_bit_counts": risk_counts,
        "base_geometry_debug": base_geometry_debug,
        "compare_frame_hist": {str(k): v for k, v in _integer_hist(sample.get("traj_compare_frame_count")).items()},
        "supervision_prefix_hist": {
            str(k): v for k, v in _integer_hist(sample.get("traj_supervision_prefix_len")).items()
        },
        "stereo_compare_frame_hist": {
            str(k): v for k, v in _integer_hist(sample.get("traj_stereo_compare_frame_count")).items()
        },
        "scalar_summaries": {
            "traj_query_fixed_view_compare_frame_count": _finite_float_summary(
                sample.get("traj_query_fixed_view_compare_frame_count")
            ),
            "traj_query_fixed_view_uv_stable_hit_count": _finite_float_summary(
                sample.get("traj_query_fixed_view_uv_stable_hit_count")
            ),
            "traj_query_fixed_view_depth_jump_hit_count": _finite_float_summary(
                sample.get("traj_query_fixed_view_depth_jump_hit_count")
            ),
            "traj_query_fixed_view_depth_anomaly_hit_count": _finite_float_summary(
                sample.get("traj_query_fixed_view_depth_anomaly_hit_count")
            ),
            "traj_query_fixed_view_max_depth_delta_m": _finite_float_summary(
                sample.get("traj_query_fixed_view_max_depth_delta_m")
            ),
            "traj_query_fixed_view_min_uv_delta_px": _finite_float_summary(
                sample.get("traj_query_fixed_view_min_uv_delta_px")
            ),
            "traj_compare_frame_count": _finite_float_summary(sample.get("traj_compare_frame_count")),
            "traj_stable_compare_frame_count": _finite_float_summary(sample.get("traj_stable_compare_frame_count")),
            "traj_supervision_prefix_len": _finite_float_summary(sample.get("traj_supervision_prefix_len")),
            "traj_supervision_count": _finite_float_summary(sample.get("traj_supervision_count")),
            "traj_query_sampler_score": _finite_float_summary(sample.get("traj_query_sampler_score")),
            "traj_motion_extent": _finite_float_summary(sample.get("traj_motion_extent")),
            "traj_motion_extent_all_valid": _finite_float_summary(sample.get("traj_motion_extent_all_valid")),
            "traj_stereo_compare_frame_count": _finite_float_summary(sample.get("traj_stereo_compare_frame_count")),
            "traj_stereo_depth_consistency_ratio": _finite_float_summary(
                sample.get("traj_stereo_depth_consistency_ratio")
            ),
            "traj_stereo_patch_error": _finite_float_summary(sample.get("traj_stereo_patch_error")),
        },
        "figures": figures,
    }


def export_query(
    *,
    trajectory_output_dir: Path,
    output_dir: Path,
    query_frame: int,
    point_size: float,
    alpha_kept: float,
    alpha_dropped: float,
    dpi: int,
) -> dict[str, object]:
    video_name = trajectory_output_dir.name
    sample_path = _load_sample_path(trajectory_output_dir, video_name, query_frame)
    with np.load(sample_path) as raw_sample:
        present_fields = set(raw_sample.files)
    sample = normalize_sample_data(sample_path)
    keypoints = np.asarray(sample["keypoints"], dtype=np.float32)

    with SceneReader(trajectory_output_dir) as reader:
        rgb = reader.get_rgb_frame(int(sample["query_frame_index"]))
    traj_uvz = np.asarray(sample["traj_uvz"], dtype=np.float32)
    visibility = None if sample.get("visibility") is None else np.asarray(sample["visibility"], dtype=np.float32)
    base_geometry_debug = _compute_base_geometry_debug(
        traj_uvz=traj_uvz,
        visibility=visibility,
        image_width=int(rgb.shape[1]),
        image_height=int(rgb.shape[0]),
    )

    stage_counts: dict[str, int | None] = {}
    missing_stage_fields: list[str] = []
    for field_name, label in STAGE_SPECS:
        mask = _mask_or_none(sample, field_name, keypoints.shape[0], present_fields=present_fields)
        stage_counts[label] = _count_true(mask)
        if mask is None:
            missing_stage_fields.append(field_name)

    stage_index, stage_reach_hist = _compute_reach_stage(
        sample,
        keypoints.shape[0],
        present_fields=present_fields,
    )
    if np.any(stage_index < 0):
        stage_reach_hist["rejected_before_base"] = int(np.count_nonzero(stage_index < 0))

    reason_counts = _bit_count_hist(sample.get("traj_mask_reason_bits"), REASON_BIT_SPECS)
    source_counts = _bit_count_hist(sample.get("traj_query_source_bits"), SOURCE_BIT_SPECS)
    risk_counts = _bit_count_hist(sample.get("traj_query_risk_bits"), RISK_BIT_SPECS)

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_panel_path = output_dir / f"q{query_frame:05d}_stage_panels.png"
    dashboard_path = output_dir / f"q{query_frame:05d}_diagnostic_dashboard.png"
    per_query_summary_path = output_dir / f"q{query_frame:05d}_summary.json"

    _render_stage_panel_figure(
        output_path=stage_panel_path,
        rgb=rgb,
        sample=sample,
        keypoints=keypoints,
        query_frame=query_frame,
        point_size=point_size,
        alpha_kept=alpha_kept,
        alpha_dropped=alpha_dropped,
        dpi=dpi,
        present_fields=present_fields,
    )
    _render_dashboard_figure(
        output_path=dashboard_path,
        rgb=rgb,
        sample=sample,
        keypoints=keypoints,
        query_frame=query_frame,
        stage_counts=stage_counts,
        stage_reach_hist=stage_reach_hist,
        reason_counts=reason_counts,
        source_counts=source_counts,
        risk_counts=risk_counts,
        dpi=dpi,
        present_fields=present_fields,
    )

    summary = _query_summary(
        trajectory_output_dir=trajectory_output_dir,
        sample_path=sample_path,
        sample=sample,
        stage_counts=stage_counts,
        stage_reach_hist=stage_reach_hist,
        reason_counts=reason_counts,
        source_counts=source_counts,
        risk_counts=risk_counts,
        figures={
            "stage_panels": str(stage_panel_path),
            "diagnostic_dashboard": str(dashboard_path),
        },
        missing_stage_fields=missing_stage_fields,
        base_geometry_debug=base_geometry_debug,
    )
    per_query_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    trajectory_output_dir = args.trajectory_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    query_frames = parse_csv_ints(args.query_frames)

    if not trajectory_output_dir.is_dir():
        raise FileNotFoundError(f"trajectory_output_dir is not a directory: {trajectory_output_dir}")

    results = [
        export_query(
            trajectory_output_dir=trajectory_output_dir,
            output_dir=output_dir,
            query_frame=query_frame,
            point_size=float(args.point_size),
            alpha_kept=float(args.alpha_kept),
            alpha_dropped=float(args.alpha_dropped),
            dpi=int(args.dpi),
        )
        for query_frame in query_frames
    ]

    summary = {
        "trajectory_output_dir": str(trajectory_output_dir),
        "query_frames": [int(q) for q in query_frames],
        "queries": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / args.summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
