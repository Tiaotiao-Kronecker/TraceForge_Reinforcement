import ast
import os
import unittest
from pathlib import Path


_SOURCE_PATH = Path(__file__).resolve().with_name("batch_infer_press_one_button_demo.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_RESOLVE_FUNC_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_traj_filter_profile"
)
_RESOLVE_MODULE = ast.Module(body=[_RESOLVE_FUNC_AST], type_ignores=[])
_RESOLVE_NAMESPACE: dict[str, object] = {}
exec(compile(_RESOLVE_MODULE, str(_SOURCE_PATH), "exec"), _RESOLVE_NAMESPACE)
resolve_traj_filter_profile = _RESOLVE_NAMESPACE["resolve_traj_filter_profile"]


class ResolveTrajFilterProfileTests(unittest.TestCase):
    def test_auto_maps_wrist_like_camera_names_to_top95(self):
        self.assertEqual(
            resolve_traj_filter_profile("varied_camera_3", "auto"),
            "wrist_manipulator_top95",
        )
        self.assertEqual(
            resolve_traj_filter_profile("hand_camera", "auto"),
            "wrist_manipulator_top95",
        )
        self.assertEqual(
            resolve_traj_filter_profile("my_wrist_cam", "auto"),
            "wrist_manipulator_top95",
        )

    def test_auto_maps_non_wrist_cameras_to_external(self):
        self.assertEqual(
            resolve_traj_filter_profile("varied_camera_1", "auto"),
            "external",
        )

    def test_explicit_profile_bypasses_auto_mapping(self):
        self.assertEqual(
            resolve_traj_filter_profile("varied_camera_3", "wrist"),
            "wrist",
        )


if __name__ == "__main__":
    unittest.main()
