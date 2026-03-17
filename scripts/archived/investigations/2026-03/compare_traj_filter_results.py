#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import load_scene_meta, normalize_sample_data


DEFAULT_TAGS = ["basic", "standard", "strict", "none_ref"]
DEFAULT_CAMERAS = ["varied_camera_1", "varied_camera_2", "varied_camera_3"]
PRIMARY_TAGS = ["basic", "standard", "strict"]


@dataclass(frozen=True)
class SampleKey:
    episode: str
    camera_name: str
    query_frame: int


def sample_key_str(sample_key: SampleKey) -> str:
    return f"{sample_key.episode}/{sample_key.camera_name}/q{sample_key.query_frame}"


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def percentile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def normalize_visibility_array(visibility: np.ndarray | None, traj: np.ndarray) -> np.ndarray | None:
    if visibility is None:
        return None
    vis = np.asarray(visibility)
    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis.squeeze(-1)
    if vis.shape == (traj.shape[1], traj.shape[0]):
        vis = vis.T
    if vis.shape != (traj.shape[0], traj.shape[1]):
        return None
    return vis > 0.5


def compute_track_features(
    traj: np.ndarray,
    visibility: np.ndarray | None,
    *,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    valid = np.isfinite(traj).all(axis=-1)
    valid_counts = valid.sum(axis=1).astype(np.int32)

    visibility_bool = normalize_visibility_array(visibility, traj)
    visibility_ratio = np.full(traj.shape[0], np.nan, dtype=np.float32)
    if visibility_bool is not None:
        for idx in range(traj.shape[0]):
            count = int(valid_counts[idx])
            if count <= 0:
                continue
            visible_count = int((visibility_bool[idx] & valid[idx]).sum())
            visibility_ratio[idx] = visible_count / count

    depth_diff_std = np.full(traj.shape[0], np.nan, dtype=np.float32)
    out_of_bounds_ratio = np.full(traj.shape[0], np.nan, dtype=np.float32)
    u = traj[:, :, 0]
    v = traj[:, :, 1]
    z = traj[:, :, 2]
    for idx in range(traj.shape[0]):
        valid_idx = valid[idx]
        count = int(valid_counts[idx])
        if count <= 0:
            continue
        valid_depths = z[idx, valid_idx]
        if valid_depths.size <= 1:
            depth_diff_std[idx] = 0.0
        else:
            depth_diff_std[idx] = float(np.std(np.diff(valid_depths)))

        valid_u = u[idx, valid_idx]
        valid_v = v[idx, valid_idx]
        out = ((valid_u < 0) | (valid_u >= width) | (valid_v < 0) | (valid_v >= height)).astype(np.float32)
        out_of_bounds_ratio[idx] = float(out.mean()) if out.size > 0 else math.nan

    return {
        "valid_frame_count": valid_counts.astype(np.float32),
        "visibility_ratio": visibility_ratio,
        "depth_diff_std": depth_diff_std,
        "out_of_bounds_ratio": out_of_bounds_ratio,
    }


def summarize_track_feature_prefix(features: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "valid_frame_count_p50": percentile_or_nan(features["valid_frame_count"], 50),
        "valid_frame_count_p90": percentile_or_nan(features["valid_frame_count"], 90),
        "visibility_ratio_p50": percentile_or_nan(features["visibility_ratio"], 50),
        "visibility_ratio_p90": percentile_or_nan(features["visibility_ratio"], 90),
        "depth_diff_std_p50": percentile_or_nan(features["depth_diff_std"], 50),
        "depth_diff_std_p90": percentile_or_nan(features["depth_diff_std"], 90),
        "out_of_bounds_ratio_p50": percentile_or_nan(features["out_of_bounds_ratio"], 50),
        "out_of_bounds_ratio_p90": percentile_or_nan(features["out_of_bounds_ratio"], 90),
    }


def sample_row(
    *,
    tag: str,
    sample_key: SampleKey,
    sample_path: Path,
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample = normalize_sample_data(sample_path)
    traj = sample["traj_uvz"].astype(np.float32)
    traj_valid_mask = sample["traj_valid_mask"].astype(bool, copy=False)
    visibility = sample.get("visibility")
    features = compute_track_features(traj, visibility, width=width, height=height)

    raw_track_count = int(traj.shape[0])
    kept_track_count = int(traj_valid_mask.sum())
    keep_ratio = (kept_track_count / raw_track_count) if raw_track_count else math.nan

    row = {
        "tag": tag,
        "episode": sample_key.episode,
        "camera_name": sample_key.camera_name,
        "query_frame": sample_key.query_frame,
        "sample_path": str(sample_path),
        "raw_track_count": raw_track_count,
        "kept_track_count": kept_track_count,
        "keep_ratio": keep_ratio,
        **summarize_track_feature_prefix(features),
    }
    payload = {
        "sample_key": sample_key,
        "sample_path": sample_path,
        "traj_valid_mask": traj_valid_mask,
        "track_features": features,
        "raw_track_count": raw_track_count,
        "kept_track_count": kept_track_count,
        "keep_ratio": keep_ratio,
    }
    return row, payload


def collect_tag_rows(
    experiment_root: Path,
    *,
    tag: str,
    episodes: list[str] | None,
    camera_names: list[str],
) -> tuple[list[dict[str, Any]], dict[SampleKey, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[SampleKey, dict[str, Any]] = {}
    inference_root = experiment_root / tag / "inference"
    if not inference_root.is_dir():
        return rows, payloads

    episode_names = episodes if episodes else sorted(path.name for path in inference_root.iterdir() if path.is_dir())
    for episode_name in episode_names:
        for camera_name in camera_names:
            camera_dir = inference_root / episode_name / camera_name
            samples_dir = camera_dir / "samples"
            if not samples_dir.is_dir():
                continue
            meta = load_scene_meta(camera_dir) or {}
            width = int(meta.get("width", 1280))
            height = int(meta.get("height", 720))
            for sample_path in sorted(samples_dir.glob("*.npz")):
                sample = normalize_sample_data(sample_path)
                sample_key = SampleKey(
                    episode=episode_name,
                    camera_name=camera_name,
                    query_frame=int(sample["query_frame_index"]),
                )
                row, payload = sample_row(
                    tag=tag,
                    sample_key=sample_key,
                    sample_path=sample_path,
                    width=width,
                    height=height,
                )
                rows.append(row)
                payloads[sample_key] = payload
    return rows, payloads


def build_tag_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "total_raw_tracks": 0,
            "total_kept_tracks": 0,
            "overall_keep_ratio": math.nan,
            "mean_sample_keep_ratio": math.nan,
            "median_sample_keep_ratio": math.nan,
        }

    raw_tracks = np.asarray([row["raw_track_count"] for row in rows], dtype=np.float32)
    kept_tracks = np.asarray([row["kept_track_count"] for row in rows], dtype=np.float32)
    keep_ratios = np.asarray([row["keep_ratio"] for row in rows], dtype=np.float32)
    return {
        "sample_count": int(len(rows)),
        "total_raw_tracks": int(raw_tracks.sum()),
        "total_kept_tracks": int(kept_tracks.sum()),
        "overall_keep_ratio": float(kept_tracks.sum() / max(raw_tracks.sum(), 1.0)),
        "mean_sample_keep_ratio": float(np.mean(keep_ratios)),
        "median_sample_keep_ratio": float(np.median(keep_ratios)),
    }


def compare_monotonicity(tag_payloads: dict[str, dict[SampleKey, dict[str, Any]]]) -> dict[str, Any]:
    common_keys = sorted(
        set.intersection(*(set(tag_payloads[tag].keys()) for tag in PRIMARY_TAGS)),
        key=sample_key_str,
    )
    monotonic_keys: list[str] = []
    non_monotonic_keys: list[str] = []
    for key in common_keys:
        keep_basic = tag_payloads["basic"][key]["keep_ratio"]
        keep_standard = tag_payloads["standard"][key]["keep_ratio"]
        keep_strict = tag_payloads["strict"][key]["keep_ratio"]
        key_str = sample_key_str(key)
        if keep_basic >= keep_standard >= keep_strict:
            monotonic_keys.append(key_str)
        else:
            non_monotonic_keys.append(key_str)
    return {
        "matched_sample_count": len(common_keys),
        "monotonic_sample_count": len(monotonic_keys),
        "non_monotonic_samples": non_monotonic_keys,
    }


def summarize_drop_profile(
    weaker_payloads: dict[SampleKey, dict[str, Any]],
    stronger_payloads: dict[SampleKey, dict[str, Any]],
) -> dict[str, Any]:
    valid_frame_counts: list[np.ndarray] = []
    visibility_ratios: list[np.ndarray] = []
    depth_diff_stds: list[np.ndarray] = []
    out_of_bounds_ratios: list[np.ndarray] = []
    dropped_track_count = 0
    matched_sample_count = 0

    common_keys = sorted(
        set(weaker_payloads.keys()) & set(stronger_payloads.keys()),
        key=sample_key_str,
    )
    for key in common_keys:
        weaker = weaker_payloads[key]
        stronger = stronger_payloads[key]
        weaker_mask = weaker["traj_valid_mask"]
        stronger_mask = stronger["traj_valid_mask"]
        if weaker_mask.shape != stronger_mask.shape:
            continue
        dropped = weaker_mask & ~stronger_mask
        if not np.any(dropped):
            matched_sample_count += 1
            continue

        matched_sample_count += 1
        dropped_track_count += int(dropped.sum())
        features = weaker["track_features"]
        valid_frame_counts.append(features["valid_frame_count"][dropped])
        visibility_ratios.append(features["visibility_ratio"][dropped])
        depth_diff_stds.append(features["depth_diff_std"][dropped])
        out_of_bounds_ratios.append(features["out_of_bounds_ratio"][dropped])

    def concat(values: list[np.ndarray]) -> np.ndarray:
        if not values:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(values).astype(np.float32, copy=False)

    valid_frame_array = concat(valid_frame_counts)
    visibility_array = concat(visibility_ratios)
    depth_diff_array = concat(depth_diff_stds)
    out_of_bounds_array = concat(out_of_bounds_ratios)
    return {
        "matched_sample_count": matched_sample_count,
        "dropped_track_count": dropped_track_count,
        "valid_frame_count_mean": mean_or_nan(valid_frame_array),
        "visibility_ratio_mean": mean_or_nan(visibility_array),
        "depth_diff_std_mean": mean_or_nan(depth_diff_array),
        "out_of_bounds_ratio_mean": mean_or_nan(out_of_bounds_array),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def build_markdown(
    *,
    tags: list[str],
    tag_summaries: dict[str, dict[str, Any]],
    monotonicity: dict[str, Any],
    standard_drop_profile: dict[str, Any],
    strict_drop_profile: dict[str, Any],
) -> str:
    lines = [
        "# Trajectory Filter Comparison",
        "",
        "## Aggregate Keep Ratios",
        "",
        "| tag | sample_count | total_raw_tracks | total_kept_tracks | overall_keep_ratio | mean_sample_keep_ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag in tags:
        summary = tag_summaries.get(tag, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    tag,
                    str(summary.get("sample_count", 0)),
                    str(summary.get("total_raw_tracks", 0)),
                    str(summary.get("total_kept_tracks", 0)),
                    format_float(summary.get("overall_keep_ratio")),
                    format_float(summary.get("mean_sample_keep_ratio")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Monotonicity",
            "",
            f"- matched samples across basic/standard/strict: {monotonicity['matched_sample_count']}",
            f"- monotonic keep-ratio samples: {monotonicity['monotonic_sample_count']}",
        ]
    )
    if monotonicity["non_monotonic_samples"]:
        lines.append(
            "- non-monotonic samples: " + ", ".join(monotonicity["non_monotonic_samples"][:20])
        )
    else:
        lines.append("- non-monotonic samples: none")

    lines.extend(
        [
            "",
            "## Drop Profiles",
            "",
            "### standard vs basic",
            f"- dropped_track_count: {standard_drop_profile['dropped_track_count']}",
            f"- dropped valid_frame_count mean: {format_float(standard_drop_profile['valid_frame_count_mean'])}",
            f"- dropped visibility_ratio mean: {format_float(standard_drop_profile['visibility_ratio_mean'])}",
            f"- dropped depth_diff_std mean: {format_float(standard_drop_profile['depth_diff_std_mean'])}",
            f"- dropped out_of_bounds_ratio mean: {format_float(standard_drop_profile['out_of_bounds_ratio_mean'])}",
            "",
            "### strict vs standard",
            f"- dropped_track_count: {strict_drop_profile['dropped_track_count']}",
            f"- dropped valid_frame_count mean: {format_float(strict_drop_profile['valid_frame_count_mean'])}",
            f"- dropped visibility_ratio mean: {format_float(strict_drop_profile['visibility_ratio_mean'])}",
            f"- dropped depth_diff_std mean: {format_float(strict_drop_profile['depth_diff_std_mean'])}",
            f"- dropped out_of_bounds_ratio mean: {format_float(strict_drop_profile['out_of_bounds_ratio_mean'])}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trajectory-filter benchmark outputs.")
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--tags", type=str, default=",".join(DEFAULT_TAGS))
    parser.add_argument("--episodes", type=str, default="")
    parser.add_argument("--camera_names", type=str, default=",".join(DEFAULT_CAMERAS))
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    tags = parse_csv_items(args.tags)
    episodes = parse_csv_items(args.episodes) or None
    camera_names = parse_csv_items(args.camera_names)

    all_rows: list[dict[str, Any]] = []
    tag_payloads: dict[str, dict[SampleKey, dict[str, Any]]] = {}
    tag_summaries: dict[str, dict[str, Any]] = {}

    for tag in tags:
        rows, payloads = collect_tag_rows(
            experiment_root,
            tag=tag,
            episodes=episodes,
            camera_names=camera_names,
        )
        all_rows.extend(rows)
        tag_payloads[tag] = payloads
        tag_summary = build_tag_summary(rows)
        tag_summaries[tag] = tag_summary
        summary_dir = experiment_root / tag / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "per_tag_summary.json").write_text(
            json.dumps(tag_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    comparison_dir = experiment_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = comparison_dir / "per_sample_metrics.csv"
    write_csv(csv_path, all_rows)

    monotonicity = compare_monotonicity(tag_payloads)
    standard_drop_profile = summarize_drop_profile(tag_payloads.get("basic", {}), tag_payloads.get("standard", {}))
    strict_drop_profile = summarize_drop_profile(tag_payloads.get("standard", {}), tag_payloads.get("strict", {}))
    combined_summary = {
        "tags": tags,
        "episodes": episodes,
        "camera_names": camera_names,
        "tag_summaries": tag_summaries,
        "monotonicity": monotonicity,
        "standard_drop_profile": standard_drop_profile,
        "strict_drop_profile": strict_drop_profile,
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(
        tags=tags,
        tag_summaries=tag_summaries,
        monotonicity=monotonicity,
        standard_drop_profile=standard_drop_profile,
        strict_drop_profile=strict_drop_profile,
    )
    (comparison_dir / "comparison.md").write_text(markdown + "\n", encoding="utf-8")

    print(f"per_sample_metrics={csv_path}")
    print(f"comparison_summary={comparison_dir / 'summary.json'}")
    print(f"comparison_markdown={comparison_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
