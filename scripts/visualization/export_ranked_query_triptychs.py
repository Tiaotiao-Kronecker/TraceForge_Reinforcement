#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_SCRIPT = PROJECT_ROOT / "scripts" / "visualization" / "compose_query_rgb_and_trajectory_gifs.py"
SUMMARY_JSON_BASENAME = "summary.json"
SUMMARY_MD_BASENAME = "summary.md"


def parse_csv_items(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select top-ranked per-query cases from compare_traceforge_output_roots.py "
            "results and export RGB/2D/3D triptych GIFs for the chosen root."
        )
    )
    parser.add_argument("--compare_results", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--root_key",
        type=str,
        default="variant",
        choices=["baseline", "variant"],
        help="Which output root to render triptychs from.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="traj_valid_mask_jaccard",
        help="Per-sample metric key from quality_summary.per_sample_rows.",
    )
    parser.add_argument(
        "--sort_order",
        type=str,
        default="asc",
        choices=["asc", "desc"],
        help="Sort direction for selecting top_k rows.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of ranked rows to export.",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default=None,
        help="Optional comma-separated camera filter.",
    )
    parser.add_argument("--gif_fps", type=int, default=10)
    parser.add_argument("--gif_dpi", type=int, default=90)
    parser.add_argument("--max_gif_tracks", type=int, default=48)
    parser.add_argument("--max_gif_cloud_points", type=int, default=3000)
    parser.add_argument("--ply_downsample", type=int, default=4)
    parser.add_argument("--depth_min", type=float, default=0.01)
    parser.add_argument("--depth_max", type=float, default=10.0)
    parser.add_argument("--line_alpha", type=float, default=0.9)
    parser.add_argument("--line_width", type=float, default=1.2)
    parser.add_argument("--panel_height", type=int, default=360)
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_rows(
    *,
    compare_results: dict[str, Any],
    metric: str,
    sort_order: str,
    top_k: int,
    camera_names: list[str],
) -> list[dict[str, Any]]:
    rows = compare_results.get("quality_summary", {}).get("per_sample_rows", [])
    if not isinstance(rows, list):
        raise ValueError("compare_results quality_summary.per_sample_rows must be a list")

    filtered_rows: list[dict[str, Any]] = []
    allowed_cameras = set(camera_names)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get(metric) is None:
            continue
        if allowed_cameras and str(row.get("camera_name")) not in allowed_cameras:
            continue
        filtered_rows.append(dict(row))

    reverse = sort_order == "desc"
    filtered_rows.sort(
        key=lambda row: (
            float(row[metric]),
            str(row.get("episode_name")),
            str(row.get("camera_name")),
            int(row.get("query_frame", 0)),
        ),
        reverse=reverse,
    )
    return filtered_rows[: max(0, int(top_k))]


def _write_stub_query_list(stub_dir: Path, query_frames: list[int]) -> None:
    stub_dir.mkdir(parents=True, exist_ok=True)
    for path in stub_dir.glob("q*_rgb_window.gif"):
        path.unlink()
    for query_frame in query_frames:
        # compose_query_rgb_and_trajectory_gifs.py only reads filenames from this directory.
        (stub_dir / f"q{int(query_frame):05d}_rgb_window.gif").touch()


def _run_compose(
    *,
    camera_dir: Path,
    sampled_rgb_gif_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(COMPOSE_SCRIPT),
        "--camera_dir",
        str(camera_dir),
        "--sampled_rgb_gif_dir",
        str(sampled_rgb_gif_dir),
        "--output_dir",
        str(output_dir),
        "--gif_fps",
        str(int(args.gif_fps)),
        "--gif_dpi",
        str(int(args.gif_dpi)),
        "--max_gif_tracks",
        str(int(args.max_gif_tracks)),
        "--max_gif_cloud_points",
        str(int(args.max_gif_cloud_points)),
        "--ply_downsample",
        str(int(args.ply_downsample)),
        "--depth_min",
        str(float(args.depth_min)),
        "--depth_max",
        str(float(args.depth_max)),
        "--line_alpha",
        str(float(args.line_alpha)),
        "--line_width",
        str(float(args.line_width)),
        "--panel_height",
        str(int(args.panel_height)),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return output_dir / SUMMARY_JSON_BASENAME


def build_ranked_artifacts(
    *,
    compare_results: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    root_payload = compare_results.get(args.root_key)
    if not isinstance(root_payload, dict) or not root_payload.get("root"):
        raise ValueError(f"compare_results missing {args.root_key}.root")

    root_dir = Path(str(root_payload["root"])).resolve()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selected_rows:
        key = (str(row["episode_name"]), str(row["camera_name"]))
        grouped.setdefault(key, []).append(row)

    artifact_rows: list[dict[str, Any]] = []
    selected_lists_root = args.output_dir / "selected_query_lists"

    for (episode_name, camera_name), rows in sorted(grouped.items()):
        query_frames = sorted(int(row["query_frame"]) for row in rows)
        stub_dir = selected_lists_root / f"{episode_name}_{camera_name}"
        _write_stub_query_list(stub_dir, query_frames)

        camera_dir = root_dir / episode_name / camera_name
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"Missing camera_dir for export: {camera_dir}")
        triptych_output_dir = args.output_dir / f"{episode_name}_{camera_name}_triptych"
        triptych_summary_path = _run_compose(
            camera_dir=camera_dir,
            sampled_rgb_gif_dir=stub_dir,
            output_dir=triptych_output_dir,
            args=args,
        )
        triptych_summary = load_json(triptych_summary_path)
        artifacts_by_query = {
            int(item["query_frame"]): item
            for item in triptych_summary.get("artifacts", [])
            if isinstance(item, dict) and item.get("query_frame") is not None
        }
        for rank_row in rows:
            query_frame = int(rank_row["query_frame"])
            artifact = artifacts_by_query.get(query_frame)
            artifact_rows.append(
                {
                    "episode_name": episode_name,
                    "camera_name": camera_name,
                    "query_frame": query_frame,
                    "metric_name": args.metric,
                    "metric_value": float(rank_row[args.metric]),
                    "selected_root_key": args.root_key,
                    "selected_root_dir": str(root_dir),
                    "triptych_output_dir": str(triptych_output_dir),
                    "triptych_summary_path": str(triptych_summary_path),
                    "query_stub_dir": str(stub_dir),
                    "composite_gif_path": (
                        artifact.get("composite_gif_path")
                        if isinstance(artifact, dict)
                        else None
                    ),
                    "tracks_2d_gif_path": (
                        artifact.get("tracks_2d_gif_path")
                        if isinstance(artifact, dict)
                        else None
                    ),
                    "tracks_3d_gif_path": (
                        artifact.get("tracks_3d_gif_path")
                        if isinstance(artifact, dict)
                        else None
                    ),
                    "rgb_gif_path": (
                        artifact.get("rgb_gif_path")
                        if isinstance(artifact, dict)
                        else None
                    ),
                }
            )

    metric_reverse = args.sort_order == "desc"
    artifact_rows.sort(
        key=lambda row: (
            float(row["metric_value"]),
            str(row["episode_name"]),
            str(row["camera_name"]),
            int(row["query_frame"]),
        ),
        reverse=metric_reverse,
    )
    for rank, row in enumerate(artifact_rows, start=1):
        row["rank"] = int(rank)
    return artifact_rows


def write_markdown_summary(*, args: argparse.Namespace, summary: dict[str, Any], summary_path: Path) -> None:
    rows = summary.get("artifacts", [])
    lines = [
        "# Ranked Query Triptych Export",
        "",
        f"- Compare results: `{summary['compare_results_path']}`",
        f"- Render root: `{summary['root_key']}` -> `{summary['root_dir']}`",
        f"- Metric: `{summary['metric']}`",
        f"- Sort order: `{summary['sort_order']}`",
        f"- Requested top_k: `{summary['top_k']}`",
        f"- Exported rows: `{len(rows)}`",
    ]
    if summary.get("camera_names"):
        lines.append(f"- Camera filter: `{','.join(summary['camera_names'])}`")
    lines.extend(
        [
            "",
            "| Rank | Episode | Camera | Query | Metric | Composite GIF |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {rank} | {episode} | {camera} | {query} | {metric_value:.6f} | `{gif}` |".format(
                rank=int(row["rank"]),
                episode=row["episode_name"],
                camera=row["camera_name"],
                query=int(row["query_frame"]),
                metric_value=float(row["metric_value"]),
                gif=row.get("composite_gif_path") or "n/a",
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.compare_results = args.compare_results.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compare_results = load_json(args.compare_results)
    camera_names = parse_csv_items(args.camera_names)
    selected_rows = select_rows(
        compare_results=compare_results,
        metric=str(args.metric),
        sort_order=str(args.sort_order),
        top_k=max(0, int(args.top_k)),
        camera_names=camera_names,
    )
    artifacts = build_ranked_artifacts(
        compare_results=compare_results,
        selected_rows=selected_rows,
        args=args,
    )

    root_payload = compare_results[args.root_key]
    summary = {
        "compare_results_path": str(args.compare_results),
        "output_dir": str(args.output_dir),
        "root_key": str(args.root_key),
        "root_dir": str(Path(str(root_payload["root"])).resolve()),
        "metric": str(args.metric),
        "sort_order": str(args.sort_order),
        "top_k": int(args.top_k),
        "camera_names": camera_names,
        "artifacts": artifacts,
    }
    summary_json_path = args.output_dir / SUMMARY_JSON_BASENAME
    summary_md_path = args.output_dir / SUMMARY_MD_BASENAME
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_summary(args=args, summary=summary, summary_path=summary_md_path)
    print(f"summary_json={summary_json_path}")
    print(f"summary_md={summary_md_path}")


if __name__ == "__main__":
    main()
