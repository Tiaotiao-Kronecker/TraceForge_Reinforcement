#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.batch_inference.adapters import ADAPTER_NAMES, get_adapter


SUPPORTED_DATASET_ADAPTERS = ("sim_file_layout", "xperience_raw")
if tuple(ADAPTER_NAMES) != SUPPORTED_DATASET_ADAPTERS:
    raise RuntimeError(
        f"Adapter registry mismatch: expected {SUPPORTED_DATASET_ADAPTERS}, got {tuple(ADAPTER_NAMES)}"
    )


def _build_dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified TraceForge batch inference entrypoint",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show dispatcher or adapter-specific help")
    parser.add_argument(
        "--dataset_adapter",
        type=str,
        choices=list(SUPPORTED_DATASET_ADAPTERS),
        help=(
            "Dataset adapter to run. sim_file_layout reads the maintained simulation episode layout; "
            "xperience_raw reads raw Xperience episodes directly."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_dispatch_parser()
    dispatch_args, remaining = parser.parse_known_args(argv)
    if dispatch_args.dataset_adapter is None:
        if dispatch_args.help:
            parser.print_help()
            return 0
        parser.error("--dataset_adapter is required")
    adapter = get_adapter(dispatch_args.dataset_adapter)
    adapter_parser = adapter.build_parser()
    if dispatch_args.help and "--help" not in remaining and "-h" not in remaining:
        remaining = ["--help", *remaining]
    parsed_args = adapter_parser.parse_args(remaining)
    try:
        finalized_args = adapter.finalize_args(parsed_args)
    except ValueError as exc:
        adapter_parser.error(str(exc))
    return int(adapter.run(finalized_args))


if __name__ == "__main__":
    raise SystemExit(main())
