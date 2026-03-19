#!/usr/bin/env python3

import argparse

from _retired_entrypoint import exit_retired_entrypoint


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.parse_known_args()
    exit_retired_entrypoint(
        entrypoint="scripts/batch_inference/infer_bridge_v2.py",
        replacement=(
            "若是 button/sim episode，请使用 scripts/batch_inference/batch_infer_press_one_button_demo.py；"
            "BridgeV2 专用入口已整体移除。"
        ),
        reason="external-only 收敛后，不再维护 BridgeV2 兼容推理链路。",
    )


if __name__ == "__main__":
    main()
