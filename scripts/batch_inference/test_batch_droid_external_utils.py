import argparse
import unittest
from pathlib import Path

from scripts.batch_inference.batch_droid_external import (
    MAINTAINED_OUTPUT_LAYOUT,
    MAINTAINED_SCENE_STORAGE_MODE,
    build_infer_command,
    build_parser,
)


class BatchDroidExternalCliSurfaceTests(unittest.TestCase):
    def test_parser_removes_legacy_output_flags(self):
        parser = build_parser()
        cli_flags = {flag for action in parser._actions for flag in action.option_strings}

        self.assertNotIn("--output_layout", cli_flags)
        self.assertNotIn("--scene_storage_mode", cli_flags)

    def test_build_infer_command_pins_maintained_output_mode(self):
        cmd = build_infer_command(
            python_bin="/usr/bin/python3",
            infer_script="/tmp/infer.py",
            geom_path=Path("/tmp/geom.npz"),
            depth_path=Path("/tmp/depth"),
            video_path=Path("/tmp/rgb"),
            trajectory_root=Path("/tmp/out"),
            camera_name="hand_camera",
            gpu_id=3,
            args=argparse.Namespace(
                frame_drop_rate=5,
                grid_size=80,
                save_visibility=False,
            ),
        )

        self.assertIn("--output_layout", cmd)
        self.assertIn("--scene_storage_mode", cmd)
        self.assertEqual(cmd[cmd.index("--output_layout") + 1], MAINTAINED_OUTPUT_LAYOUT)
        self.assertEqual(cmd[cmd.index("--scene_storage_mode") + 1], MAINTAINED_SCENE_STORAGE_MODE)


if __name__ == "__main__":
    unittest.main()
