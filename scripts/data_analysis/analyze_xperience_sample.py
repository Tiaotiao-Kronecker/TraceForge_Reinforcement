#!/usr/bin/env python3
"""
Summarize the Xperience-10M sample dataset into small JSON / TSV artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.data_analysis.xperience_sample_utils import (
    XperienceSampleDataset,
    resolve_dataset_dir,
    summarize_dataset_dir,
    write_schema_tsv,
)


DEFAULT_OUTPUT_DIR = Path("data_tmp/xperience_sample_analysis")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the Xperience-10M sample dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root that contains annotation.hdf5 and MP4 files. Defaults to $XPERIENCE_SAMPLE_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write summary artifacts.",
    )
    parser.add_argument(
        "--probe-frame",
        type=int,
        default=None,
        help="Optional frame index for one detailed per-frame JSON dump. Defaults to the midpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, schema = summarize_dataset_dir(dataset_dir)
    summary_path = output_dir / "summary.json"
    schema_path = output_dir / "annotation_schema.tsv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_schema_tsv(schema, schema_path)

    with XperienceSampleDataset(dataset_dir) as dataset:
        probe_index = dataset.validate_index(args.probe_frame if args.probe_frame is not None else len(dataset) // 2)
        sample = dataset.get_frame(
            probe_index,
            video_streams=("stereo_left", "stereo_right", "fisheye_cam1"),
            load_video=False,
            load_depth=True,
            load_mocap=True,
            load_imu=True,
            imu_radius=12,
        )

    probe_path = output_dir / f"frame_probe_{probe_index:04d}.json"
    probe_path.write_text(json.dumps(sample.summary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = {
        "summary": str(summary_path),
        "schema": str(schema_path),
        "frame_probe": str(probe_path),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
