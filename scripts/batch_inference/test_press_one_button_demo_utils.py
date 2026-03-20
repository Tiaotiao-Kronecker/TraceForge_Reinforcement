import ast
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

_PARSE_ARGS_FUNC_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
)


def _collect_cli_flags(func_ast: ast.FunctionDef) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value)
    return flags


_CLI_FLAGS = _collect_cli_flags(_PARSE_ARGS_FUNC_AST)


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


class PressOneButtonCliSurfaceTests(unittest.TestCase):
    def test_removes_dead_batch_only_legacy_flags(self):
        self.assertNotIn("--keyframes_per_sec", _CLI_FLAGS)
        self.assertNotIn("--horizon", _CLI_FLAGS)
        self.assertNotIn("--frame_drop_rate", _CLI_FLAGS)
        self.assertNotIn("--max_frames_per_video", _CLI_FLAGS)
        self.assertNotIn("--depth_volatility_mode", _CLI_FLAGS)

    def test_exposes_query_prefilter_and_support_ratio_flags(self):
        self.assertIn("--query_prefilter_mode", _CLI_FLAGS)
        self.assertIn("--query_prefilter_wrist_rank_keep_ratio", _CLI_FLAGS)
        self.assertIn("--support_grid_ratio", _CLI_FLAGS)


if __name__ == "__main__":
    unittest.main()
