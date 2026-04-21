from __future__ import annotations

from scripts.batch_inference import batch_infer_sim_file_layout as runner


ADAPTER_NAME = "sim_file_layout"


def build_parser():
    return runner.build_parser()


def finalize_args(args):
    return runner.finalize_args(args)


def run(args) -> int:
    return runner.run(args)
