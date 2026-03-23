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
_CAMERA_TASK_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.ClassDef) and node.name == "CameraTask"
)
_SAFE_PER_QUERY_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "_safe_per_query_seconds"
)
_BUILD_TASK_RECORD_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "build_camera_task_metric_record"
)
_BUILD_BATCH_SUMMARY_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "build_batch_run_summary"
)
_TELEMETRY_MODULE = ast.Module(
    body=[
        _CAMERA_TASK_AST,
        _SAFE_PER_QUERY_AST,
        _BUILD_TASK_RECORD_AST,
        _BUILD_BATCH_SUMMARY_AST,
    ],
    type_ignores=[],
)
_TELEMETRY_GLOBALS = {
    "dataclass": __import__("dataclasses").dataclass,
    "Path": Path,
    "argparse": __import__("argparse"),
    "Any": object,
}
exec(compile(_TELEMETRY_MODULE, str(_SOURCE_PATH), "exec"), _TELEMETRY_GLOBALS)
CameraTask = _TELEMETRY_GLOBALS["CameraTask"]
build_camera_task_metric_record = _TELEMETRY_GLOBALS["build_camera_task_metric_record"]
build_batch_run_summary = _TELEMETRY_GLOBALS["build_batch_run_summary"]


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


def _collect_cli_defaults(func_ast: ast.FunctionDef) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flag = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flag = arg.value
                break
        if flag is None:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                defaults[flag] = keyword.value.value
    return defaults


_CLI_FLAGS = _collect_cli_flags(_PARSE_ARGS_FUNC_AST)
_CLI_DEFAULTS = _collect_cli_defaults(_PARSE_ARGS_FUNC_AST)


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

    def test_exposes_episode_names_file_filter_flag(self):
        self.assertIn("--episode_names_file", _CLI_FLAGS)
        self.assertIsNone(_CLI_DEFAULTS.get("--episode_names_file"))

    def test_exposes_query_prefilter_and_support_ratio_flags(self):
        self.assertIn("--query_prefilter_mode", _CLI_FLAGS)
        self.assertIn("--query_prefilter_wrist_rank_keep_ratio", _CLI_FLAGS)
        self.assertIn("--support_grid_ratio", _CLI_FLAGS)
        self.assertIn("--traj_filter_ablation_mode", _CLI_FLAGS)

    def test_num_iters_default_is_five(self):
        self.assertEqual(_CLI_DEFAULTS.get("--num_iters"), 5)


class BatchTelemetryRecordTests(unittest.TestCase):
    def test_build_camera_task_metric_record_computes_per_query_fields(self):
        task = CameraTask(
            task_index=1,
            total_tasks=60,
            episode_dir=Path("/tmp/episode_00001_green"),
            out_episode_dir=Path("/tmp/output/episode_00001_green"),
            camera_name="varied_camera_1",
            query_frame_schedule_path=Path("/tmp/output/episode_00001_green/_shared/schedule.json"),
        )
        args = type(
            "Args",
            (),
            {
                "device": "cuda:0",
                "num_iters": 5,
                "traj_filter_profile": "external",
            },
        )()

        record = build_camera_task_metric_record(
            task=task,
            gpu_id=0,
            args=args,
            query_frame_count=10,
            process_seconds=50.0,
            save_seconds=2.0,
            started_at_unix=1000.0,
            finished_at_unix=1052.0,
            status="success",
            retryable_cuda_error=False,
            error_message=None,
        )

        self.assertEqual(record["episode_name"], "episode_00001_green")
        self.assertEqual(record["camera_name"], "varied_camera_1")
        self.assertEqual(record["num_iters"], 5)
        self.assertEqual(record["query_frame_count"], 10)
        self.assertAlmostEqual(record["process_seconds_per_query"], 5.0)
        self.assertAlmostEqual(record["save_seconds_per_query"], 0.2)
        self.assertAlmostEqual(record["total_seconds_per_query"], 5.2)
        self.assertEqual(record["status"], "success")

    def test_build_batch_run_summary_preserves_fixed_keyframe_config(self):
        args = type(
            "Args",
            (),
            {
                "camera_names": ["varied_camera_1", "varied_camera_3"],
                "num_iters": 6,
                "keyframe_seed": 0,
                "keyframes_per_sec_min": 5,
                "keyframes_per_sec_max": 5,
                "fps": 1,
                "max_num_frames": 512,
                "future_len": 32,
                "grid_size": 80,
                "support_grid_ratio": 0.8,
                "filter_level": "standard",
                "traj_filter_profile": "auto",
                "external_geom_name": "trajectory_valid.h5",
                "external_extr_mode": "w2c",
            },
        )()

        summary = build_batch_run_summary(
            args=args,
            base_path=Path("/data2/test"),
            out_dir=Path("/data2/out"),
            gpu_ids=[0, 1, 2, 3],
            episode_count=30,
            camera_task_count=60,
            total_camera_success=60,
            total_camera_fail=0,
            wall_clock_seconds=1234.5,
        )

        self.assertEqual(summary["camera_names"], ["varied_camera_1", "varied_camera_3"])
        self.assertEqual(summary["gpu_ids"], [0, 1, 2, 3])
        self.assertEqual(summary["episode_count"], 30)
        self.assertEqual(summary["camera_task_count"], 60)
        self.assertEqual(summary["keyframes_per_sec_min"], 5)
        self.assertEqual(summary["keyframes_per_sec_max"], 5)
        self.assertAlmostEqual(summary["support_grid_ratio"], 0.8)
        self.assertAlmostEqual(summary["wall_clock_seconds"], 1234.5)


if __name__ == "__main__":
    unittest.main()
