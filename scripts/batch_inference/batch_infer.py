#!/usr/bin/env python3

import argparse

from _retired_entrypoint import exit_retired_entrypoint


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.parse_known_args()
    exit_retired_entrypoint(
        entrypoint="scripts/batch_inference/batch_infer.py",
        replacement=(
            "scripts/batch_inference/batch_infer_press_one_button_demo.py "
            "(button/sim episodes) 或 scripts/batch_inference/batch_droid_external.py (DROID)"
        ),
        reason="仓库已收敛为 external-only，旧通用批处理入口不再维护，也不再支持无外部几何的流程。",
    )


if __name__ == "__main__":
    main()
