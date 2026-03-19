#!/usr/bin/env python3

import argparse

from _retired_entrypoint import exit_retired_entrypoint


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.parse_known_args()
    exit_retired_entrypoint(
        entrypoint="scripts/batch_inference/batch_droid.py",
        replacement="scripts/batch_inference/batch_droid_external.py",
        reason="仓库已收敛为 external-only，DROID 的纯 RGB/VGGT 批处理入口不再维护。",
    )


if __name__ == "__main__":
    main()
