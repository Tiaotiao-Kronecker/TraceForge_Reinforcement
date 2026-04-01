#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SUMMARY_BASENAME = "_batch_run_summary.json"
TASK_METRICS_BASENAME = "_camera_task_metrics.jsonl"
TASK_PROFILES_BASENAME = "_camera_task_profiles.jsonl"
HARDWARE_TELEMETRY_BASENAME = "_hardware_telemetry.jsonl"

DEFAULT_PROCESS_PROFILE_KEYS = (
    "load_rgb_seconds",
    "load_depth_seconds",
    "depth_pose_wrapper_seconds",
    "prepare_depth_filter_seconds",
    "prepare_inputs_seconds",
    "tracker_inference_total_seconds",
    "tracker_model_forward_seconds",
    "filter_eval_seconds",
    "process_other_seconds",
    "process_total_seconds",
)
DEFAULT_SAVE_PROFILE_KEYS = (
    "prepare_bundles_seconds",
    "filter_eval_seconds",
    "sample_write_seconds",
    "scene_meta_write_seconds",
    "query_frame_save_loop_seconds",
    "save_other_seconds",
    "save_total_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize batch inference telemetry from "
            "_batch_run_summary.json, _camera_task_metrics.jsonl, and optional profile/hardware JSONLs."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="TraceForge batch output root that contains telemetry files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the structured analysis JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path to write the Markdown report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    rank = (len(sorted_values) - 1) * q
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = rank - lower
    return float(lower_value * (1.0 - weight) + upper_value * weight)


def is_timing_profile_key(key: str, *, profile_keys: tuple[str, ...]) -> bool:
    return key in profile_keys or key.endswith("_seconds")


def summarize_run_overview(
    summary: dict[str, Any],
    task_records: list[dict[str, Any]],
) -> dict[str, Any]:
    success_records = [record for record in task_records if record.get("status") == "success"]
    total_query_count = int(
        sum(int(record.get("query_frame_count") or 0) for record in success_records)
    )
    wall_clock_seconds = float(summary.get("wall_clock_seconds") or 0.0)
    telemetry_gpu_ids = summary.get("telemetry_gpu_ids") or summary.get("gpu_ids") or []
    physical_gpu_count = len(telemetry_gpu_ids)
    if physical_gpu_count <= 0 and total_query_count > 0:
        physical_gpu_count = 1

    total_task_seconds = float(
        sum(float(record.get("total_seconds") or 0.0) for record in success_records)
    )
    total_process_seconds = float(
        sum(float(record.get("process_seconds") or 0.0) for record in success_records)
    )
    total_save_seconds = float(
        sum(float(record.get("save_seconds") or 0.0) for record in success_records)
    )

    worker_slot_count_raw = summary.get("worker_slot_count")
    return {
        "task_count": int(len(task_records)),
        "success_task_count": int(len(success_records)),
        "failed_task_count": int(len(task_records) - len(success_records)),
        "total_query_count": total_query_count,
        "wall_clock_seconds": wall_clock_seconds,
        "physical_gpu_count": int(physical_gpu_count),
        "worker_slot_count": (
            int(worker_slot_count_raw)
            if worker_slot_count_raw not in (None, 0)
            else None
        ),
        "cluster_seconds_per_query": safe_div(wall_clock_seconds, float(total_query_count)),
        "single_gpu_seconds_per_query": safe_div(
            wall_clock_seconds * float(physical_gpu_count),
            float(total_query_count),
        ),
        "slot_seconds_per_query": safe_div(total_task_seconds, float(total_query_count)),
        "process_slot_seconds_per_query": safe_div(total_process_seconds, float(total_query_count)),
        "save_slot_seconds_per_query": safe_div(total_save_seconds, float(total_query_count)),
    }


def summarize_task_groups(
    task_records: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in task_records:
        key = tuple(record.get(field) for field in group_fields)
        if any(value is None for value in key):
            continue
        grouped.setdefault(key, []).append(record)

    summaries: list[dict[str, Any]] = []
    for key, records in grouped.items():
        success_records = [record for record in records if record.get("status") == "success"]
        query_count = int(sum(int(record.get("query_frame_count") or 0) for record in success_records))
        total_seconds = float(sum(float(record.get("total_seconds") or 0.0) for record in success_records))
        process_seconds = float(sum(float(record.get("process_seconds") or 0.0) for record in success_records))
        save_seconds = float(sum(float(record.get("save_seconds") or 0.0) for record in success_records))
        per_query_values = [
            float(record["total_seconds_per_query"])
            for record in success_records
            if record.get("total_seconds_per_query") is not None
        ]
        payload = {
            "group_label": " / ".join(str(value) for value in key),
            "task_count": int(len(records)),
            "success_task_count": int(len(success_records)),
            "failed_task_count": int(len(records) - len(success_records)),
            "query_count": query_count,
            "slot_seconds_per_query": safe_div(total_seconds, float(query_count)),
            "process_slot_seconds_per_query": safe_div(process_seconds, float(query_count)),
            "save_slot_seconds_per_query": safe_div(save_seconds, float(query_count)),
            "task_total_seconds_per_query_p50": percentile(per_query_values, 0.50),
            "task_total_seconds_per_query_p95": percentile(per_query_values, 0.95),
        }
        for field, value in zip(group_fields, key):
            payload[field] = value
        summaries.append(payload)

    return sorted(
        summaries,
        key=lambda record: (
            -(record.get("query_count") or 0),
            record["group_label"],
        ),
    )


def summarize_profile_records(
    profile_records: list[dict[str, Any]],
    *,
    profile_field: str,
    profile_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    success_records = [record for record in profile_records if record.get("status") == "success"]
    total_query_count = int(
        sum(int(record.get("query_frame_count") or 0) for record in success_records)
    )
    if total_query_count <= 0:
        return []

    totals: dict[str, float] = {}
    ordered_profile_keys = tuple(dict.fromkeys(profile_keys))
    for record in success_records:
        stats = record.get(profile_field) or {}
        if not isinstance(stats, dict):
            continue
        for key, value in stats.items():
            if not is_timing_profile_key(key, profile_keys=ordered_profile_keys):
                continue
            totals[key] = float(totals.get(key, 0.0) + float(value or 0.0))

    ordered_key_set = set(ordered_profile_keys)
    ordered_keys = list(ordered_profile_keys) + [
        key for key in sorted(totals.keys()) if key not in ordered_key_set
    ]
    rows: list[dict[str, Any]] = []
    for key in ordered_keys:
        if key not in totals:
            continue
        rows.append(
            {
                "profile_key": key,
                "total_seconds": float(totals[key]),
                "seconds_per_query": safe_div(float(totals[key]), float(total_query_count)),
            }
        )
    return rows


def summarize_hardware_samples(
    hardware_records: list[dict[str, Any]],
) -> dict[str, Any]:
    gpu_by_id: dict[int, dict[str, list[float]]] = {}
    cpu_iowait_values: list[float] = []
    disk_read_values: list[float] = []
    disk_write_values: list[float] = []

    for record in hardware_records:
        for gpu_sample in record.get("gpu_samples") or []:
            gpu_id = int(gpu_sample["gpu_id"])
            bucket = gpu_by_id.setdefault(
                gpu_id,
                {
                    "utilization_gpu_pct": [],
                    "utilization_memory_pct": [],
                    "memory_used_mib": [],
                    "power_draw_watts": [],
                    "name": [gpu_sample.get("name")],
                },
            )
            for metric_key in (
                "utilization_gpu_pct",
                "utilization_memory_pct",
                "memory_used_mib",
                "power_draw_watts",
            ):
                value = gpu_sample.get(metric_key)
                if value is not None:
                    bucket[metric_key].append(float(value))

        cpu_io_metrics = record.get("cpu_io_metrics") or {}
        if cpu_io_metrics.get("cpu_iowait_pct") is not None:
            cpu_iowait_values.append(float(cpu_io_metrics["cpu_iowait_pct"]))
        if cpu_io_metrics.get("disk_read_bytes_per_sec") is not None:
            disk_read_values.append(float(cpu_io_metrics["disk_read_bytes_per_sec"]))
        if cpu_io_metrics.get("disk_write_bytes_per_sec") is not None:
            disk_write_values.append(float(cpu_io_metrics["disk_write_bytes_per_sec"]))

    gpu_rows: list[dict[str, Any]] = []
    for gpu_id, metrics in sorted(gpu_by_id.items()):
        gpu_rows.append(
            {
                "gpu_id": gpu_id,
                "name": next((value for value in metrics["name"] if value is not None), None),
                "utilization_gpu_pct_mean": (
                    statistics.mean(metrics["utilization_gpu_pct"])
                    if metrics["utilization_gpu_pct"]
                    else None
                ),
                "utilization_gpu_pct_max": (
                    max(metrics["utilization_gpu_pct"])
                    if metrics["utilization_gpu_pct"]
                    else None
                ),
                "utilization_memory_pct_mean": (
                    statistics.mean(metrics["utilization_memory_pct"])
                    if metrics["utilization_memory_pct"]
                    else None
                ),
                "memory_used_gib_mean": (
                    statistics.mean(metrics["memory_used_mib"]) / 1024.0
                    if metrics["memory_used_mib"]
                    else None
                ),
                "power_draw_watts_mean": (
                    statistics.mean(metrics["power_draw_watts"])
                    if metrics["power_draw_watts"]
                    else None
                ),
            }
        )

    return {
        "sample_count": int(len(hardware_records)),
        "gpu_summary": gpu_rows,
        "cpu_iowait_pct_mean": statistics.mean(cpu_iowait_values) if cpu_iowait_values else None,
        "cpu_iowait_pct_p95": percentile(cpu_iowait_values, 0.95),
        "disk_read_mib_per_sec_mean": (
            statistics.mean(disk_read_values) / float(1024 ** 2)
            if disk_read_values
            else None
        ),
        "disk_write_mib_per_sec_mean": (
            statistics.mean(disk_write_values) / float(1024 ** 2)
            if disk_write_values
            else None
        ),
    }


def format_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        headers = [header for header, _key in columns]
        separators = ["---"] * len(columns)
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(separators) + " |",
            ]
        )

    headers = [header for header, _key in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered_values: list[str] = []
        for _header, key in columns:
            value = row.get(key)
            if value is None:
                rendered_values.append("-")
            elif isinstance(value, float):
                rendered_values.append(format_float(value))
            else:
                rendered_values.append(str(value))
        lines.append("| " + " | ".join(rendered_values) + " |")
    return "\n".join(lines)


def build_markdown_report(analysis: dict[str, Any]) -> str:
    overview = analysis["overview"]
    sections = [
        "# Batch Telemetry Summary",
        "",
        "## Overview",
        "",
        format_table(
            [
                {
                    "wall_clock_seconds": overview["wall_clock_seconds"],
                    "total_query_count": overview["total_query_count"],
                    "physical_gpu_count": overview["physical_gpu_count"],
                    "worker_slot_count": overview["worker_slot_count"],
                    "cluster_seconds_per_query": overview["cluster_seconds_per_query"],
                    "single_gpu_seconds_per_query": overview["single_gpu_seconds_per_query"],
                    "slot_seconds_per_query": overview["slot_seconds_per_query"],
                    "process_slot_seconds_per_query": overview["process_slot_seconds_per_query"],
                    "save_slot_seconds_per_query": overview["save_slot_seconds_per_query"],
                }
            ],
            [
                ("Wall(s)", "wall_clock_seconds"),
                ("Queries", "total_query_count"),
                ("GPUs", "physical_gpu_count"),
                ("Slots", "worker_slot_count"),
                ("Wall/query", "cluster_seconds_per_query"),
                ("SingleGPU/query", "single_gpu_seconds_per_query"),
                ("Slot/query", "slot_seconds_per_query"),
                ("Process/query", "process_slot_seconds_per_query"),
                ("Save/query", "save_slot_seconds_per_query"),
            ],
        ),
        "",
        "## By Camera",
        "",
        format_table(
            analysis["by_camera"],
            [
                ("Camera", "group_label"),
                ("Tasks", "task_count"),
                ("Queries", "query_count"),
                ("Slot/query", "slot_seconds_per_query"),
                ("Process/query", "process_slot_seconds_per_query"),
                ("Save/query", "save_slot_seconds_per_query"),
                ("P50", "task_total_seconds_per_query_p50"),
                ("P95", "task_total_seconds_per_query_p95"),
            ],
        ),
        "",
        "## By Camera/Profile",
        "",
        format_table(
            analysis["by_camera_profile"],
            [
                ("Camera/Profile", "group_label"),
                ("Tasks", "task_count"),
                ("Queries", "query_count"),
                ("Slot/query", "slot_seconds_per_query"),
                ("Process/query", "process_slot_seconds_per_query"),
                ("Save/query", "save_slot_seconds_per_query"),
            ],
        ),
        "",
        "## By GPU",
        "",
        format_table(
            analysis["by_gpu"],
            [
                ("GPU", "group_label"),
                ("Tasks", "task_count"),
                ("Queries", "query_count"),
                ("Slot/query", "slot_seconds_per_query"),
                ("Process/query", "process_slot_seconds_per_query"),
            ],
        ),
        "",
        "## By Worker",
        "",
        format_table(
            analysis["by_worker"],
            [
                ("Worker", "group_label"),
                ("Tasks", "task_count"),
                ("Queries", "query_count"),
                ("Slot/query", "slot_seconds_per_query"),
                ("Process/query", "process_slot_seconds_per_query"),
            ],
        ),
    ]

    if analysis["process_profile"]:
        sections.extend(
            [
                "",
                "## Process Profile",
                "",
                format_table(
                    analysis["process_profile"],
                    [
                        ("Key", "profile_key"),
                        ("Total(s)", "total_seconds"),
                        ("Sec/query", "seconds_per_query"),
                    ],
                ),
            ]
        )

    if analysis["save_profile"]:
        sections.extend(
            [
                "",
                "## Save Profile",
                "",
                format_table(
                    analysis["save_profile"],
                    [
                        ("Key", "profile_key"),
                        ("Total(s)", "total_seconds"),
                        ("Sec/query", "seconds_per_query"),
                    ],
                ),
            ]
        )

    hardware = analysis["hardware"]
    if hardware["gpu_summary"] or hardware["sample_count"] > 0:
        sections.extend(
            [
                "",
                "## Hardware",
                "",
                format_table(
                    hardware["gpu_summary"],
                    [
                        ("GPU", "gpu_id"),
                        ("Name", "name"),
                        ("GPU util mean", "utilization_gpu_pct_mean"),
                        ("GPU util max", "utilization_gpu_pct_max"),
                        ("Mem util mean", "utilization_memory_pct_mean"),
                        ("Mem used GiB", "memory_used_gib_mean"),
                        ("Power mean", "power_draw_watts_mean"),
                    ],
                ),
                "",
                format_table(
                    [
                        {
                            "sample_count": hardware["sample_count"],
                            "cpu_iowait_pct_mean": hardware["cpu_iowait_pct_mean"],
                            "cpu_iowait_pct_p95": hardware["cpu_iowait_pct_p95"],
                            "disk_read_mib_per_sec_mean": hardware["disk_read_mib_per_sec_mean"],
                            "disk_write_mib_per_sec_mean": hardware["disk_write_mib_per_sec_mean"],
                        }
                    ],
                    [
                        ("Samples", "sample_count"),
                        ("CPU iowait mean", "cpu_iowait_pct_mean"),
                        ("CPU iowait p95", "cpu_iowait_pct_p95"),
                        ("Disk read MiB/s", "disk_read_mib_per_sec_mean"),
                        ("Disk write MiB/s", "disk_write_mib_per_sec_mean"),
                    ],
                ),
            ]
        )

    return "\n".join(sections) + "\n"


def build_analysis(run_root: Path) -> dict[str, Any]:
    summary = load_json(run_root / SUMMARY_BASENAME)
    task_records = load_jsonl(run_root / TASK_METRICS_BASENAME)
    profile_records = load_jsonl(run_root / TASK_PROFILES_BASENAME)
    hardware_records = load_jsonl(run_root / HARDWARE_TELEMETRY_BASENAME)
    return {
        "summary": summary,
        "overview": summarize_run_overview(summary, task_records),
        "by_camera": summarize_task_groups(task_records, group_fields=("camera_name",)),
        "by_camera_profile": summarize_task_groups(
            task_records,
            group_fields=("camera_name", "traj_filter_profile"),
        ),
        "by_gpu": summarize_task_groups(task_records, group_fields=("gpu_id",)),
        "by_worker": summarize_task_groups(task_records, group_fields=("worker_label",)),
        "process_profile": summarize_profile_records(
            profile_records,
            profile_field="profile_stats",
            profile_keys=DEFAULT_PROCESS_PROFILE_KEYS,
        ),
        "save_profile": summarize_profile_records(
            profile_records,
            profile_field="save_profile_stats",
            profile_keys=DEFAULT_SAVE_PROFILE_KEYS,
        ),
        "hardware": summarize_hardware_samples(hardware_records),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    analysis = build_analysis(run_root)
    report = build_markdown_report(analysis)
    print(report, end="")

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.output_md is not None:
        args.output_md.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
