#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.minimal_tracker_inference.minimal_tracker_core import (
    attach_profiler_summary,
    create_synthetic_case,
    load_case_from_npz,
    load_tracker_model,
    prepare_tracker_case,
    profile_tracker_flops,
    run_tracker_once,
    summarize_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal TAPIP3D tracker runner. This path skips dataset walking, "
            "trajectory filtering, artifact writing, and batch orchestration."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--case_npz", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic_frames", type=int, default=12)
    parser.add_argument("--synthetic_height", type=int, default=256)
    parser.add_argument("--synthetic_width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query_grid_size", type=int, default=80)
    parser.add_argument("--query_frame", type=int, default=0)
    parser.add_argument("--support_grid_size", type=int, default=0)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--precision_modes", type=str, default="fp32,bf16")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=3)
    parser.add_argument("--profile_flops", action="store_true")
    parser.add_argument("--h100_variant", type=str, default="sxm", choices=["sxm", "nvl"])
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def _parse_precision_modes(raw: str) -> list[str]:
    from utils.inference_utils import normalize_tracker_precision_mode

    modes = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not modes:
        raise ValueError("precision_modes must contain at least one mode")
    return [normalize_tracker_precision_mode(item) for item in modes]


def main() -> None:
    args = parse_args()
    precision_modes = _parse_precision_modes(args.precision_modes)
    if not args.synthetic and args.case_npz is None:
        raise ValueError("Either --case_npz or --synthetic must be provided")

    if args.synthetic:
        case = create_synthetic_case(
            frames=int(args.synthetic_frames),
            height=int(args.synthetic_height),
            width=int(args.synthetic_width),
        )
    else:
        case = load_case_from_npz(args.case_npz)

    prepared_case = prepare_tracker_case(
        case,
        device=args.device,
        query_grid_size=int(args.query_grid_size),
        query_frame=int(args.query_frame),
    )
    model = load_tracker_model(args.checkpoint, device=args.device)
    seq_len = int(getattr(model, "seq_len", prepared_case.video.shape[0]))

    all_results = []
    for precision_mode in precision_modes:
        for _ in range(int(args.warmup_runs)):
            run_tracker_once(
                model=model,
                prepared_case=prepared_case,
                num_iters=int(args.num_iters),
                support_grid_size=int(args.support_grid_size),
                precision_mode=precision_mode,
            )

        timed_runs = []
        for _ in range(int(args.benchmark_runs)):
            timed_runs.append(
                run_tracker_once(
                    model=model,
                    prepared_case=prepared_case,
                    num_iters=int(args.num_iters),
                    support_grid_size=int(args.support_grid_size),
                    precision_mode=precision_mode,
                )
            )

        best_run = min(timed_runs, key=lambda item: item.wall_time_seconds)
        if args.profile_flops:
            profiled_flops, top_profiler_ops = profile_tracker_flops(
                model=model,
                prepared_case=prepared_case,
                num_iters=int(args.num_iters),
                support_grid_size=int(args.support_grid_size),
                precision_mode=precision_mode,
            )
            best_run = attach_profiler_summary(
                best_run,
                profiled_flops=profiled_flops,
                top_profiler_ops=top_profiler_ops,
                h100_variant=args.h100_variant,
            )
        all_results.append(best_run)

    summary = summarize_run(
        case_name=case.name,
        checkpoint_path=args.checkpoint,
        prepared_case=prepared_case,
        num_iters=int(args.num_iters),
        support_grid_size=int(args.support_grid_size),
        seq_len=seq_len,
        run_results=all_results,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
