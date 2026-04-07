import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SOURCE_PATH = Path(__file__).resolve().with_name("batch_infer_press_one_button_demo.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_RESOLVE_FUNC_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_traj_filter_profile"
)
_RESOLVE_MODULE = ast.Module(body=[_RESOLVE_FUNC_AST], type_ignores=[])
_RESOLVE_NAMESPACE: dict[str, object] = {}
exec(compile(_RESOLVE_MODULE, str(_SOURCE_PATH), "exec"), _RESOLVE_NAMESPACE)
resolve_traj_filter_profile = _RESOLVE_NAMESPACE["resolve_traj_filter_profile"]

_HAS_FILES_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "_has_files"
)
_FIND_VALID_EPISODES_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "find_valid_episodes"
)
_FIND_VALID_EPISODES_MODULE = ast.Module(
    body=[_HAS_FILES_AST, _FIND_VALID_EPISODES_AST],
    type_ignores=[],
)
_FIND_VALID_EPISODES_NAMESPACE = {
    "Path": Path,
}
exec(compile(_FIND_VALID_EPISODES_MODULE, str(_SOURCE_PATH), "exec"), _FIND_VALID_EPISODES_NAMESPACE)
find_valid_episodes = _FIND_VALID_EPISODES_NAMESPACE["find_valid_episodes"]

_PARSE_ARGS_FUNC_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
)
_PARSE_CAMERA_NAMES_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_camera_names"
)
_PARSE_CAMERA_INT_OVERRIDES_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_camera_int_overrides"
)
_RESOLVE_SCHEDULE_CAMERA_NAMES_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "resolve_schedule_camera_names"
)
_RESOLVE_CAMERA_NUM_ITERS_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_camera_num_iters"
)
_BUILD_CAMERA_ARGS_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "build_camera_args"
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
_BUILD_TASK_PROFILE_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "build_camera_task_profile_record"
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
        _BUILD_TASK_PROFILE_AST,
        _BUILD_BATCH_SUMMARY_AST,
    ],
    type_ignores=[],
)
_TELEMETRY_GLOBALS = {
    "dataclass": __import__("dataclasses").dataclass,
    "Path": Path,
    "argparse": __import__("argparse"),
    "Any": object,
    "_DEFAULT_DEPTH_FILTER_WORKERS": 8,
}
exec(compile(_TELEMETRY_MODULE, str(_SOURCE_PATH), "exec"), _TELEMETRY_GLOBALS)
CameraTask = _TELEMETRY_GLOBALS["CameraTask"]
build_camera_task_metric_record = _TELEMETRY_GLOBALS["build_camera_task_metric_record"]
build_camera_task_profile_record = _TELEMETRY_GLOBALS["build_camera_task_profile_record"]
build_batch_run_summary = _TELEMETRY_GLOBALS["build_batch_run_summary"]

_CAMERA_ARGS_MODULE = ast.Module(
    body=[
        _PARSE_CAMERA_NAMES_AST,
        _RESOLVE_SCHEDULE_CAMERA_NAMES_AST,
        _PARSE_CAMERA_INT_OVERRIDES_AST,
        _RESOLVE_FUNC_AST,
        _RESOLVE_CAMERA_NUM_ITERS_AST,
        _BUILD_CAMERA_ARGS_AST,
    ],
    type_ignores=[],
)
_CAMERA_ARGS_GLOBALS = {
    "Path": Path,
    "argparse": __import__("argparse"),
    "copy": __import__("copy"),
}
exec(compile(_CAMERA_ARGS_MODULE, str(_SOURCE_PATH), "exec"), _CAMERA_ARGS_GLOBALS)
parse_camera_int_overrides = _CAMERA_ARGS_GLOBALS["parse_camera_int_overrides"]
resolve_schedule_camera_names = _CAMERA_ARGS_GLOBALS["resolve_schedule_camera_names"]
resolve_camera_num_iters = _CAMERA_ARGS_GLOBALS["resolve_camera_num_iters"]
build_camera_args = _CAMERA_ARGS_GLOBALS["build_camera_args"]


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


def _collect_cli_choices(func_ast: ast.FunctionDef) -> dict[str, tuple[str, ...]]:
    choices: dict[str, tuple[str, ...]] = {}
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
            if keyword.arg != "choices":
                continue
            if not isinstance(keyword.value, (ast.List, ast.Tuple)):
                continue
            values: list[str] = []
            for element in keyword.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    values.append(element.value)
            if values:
                choices[flag] = tuple(values)
    return choices


_CLI_FLAGS = _collect_cli_flags(_PARSE_ARGS_FUNC_AST)
_CLI_DEFAULTS = _collect_cli_defaults(_PARSE_ARGS_FUNC_AST)
_CLI_CHOICES = _collect_cli_choices(_PARSE_ARGS_FUNC_AST)


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

    def test_pick_place_profile_only_applies_to_wrist_like_cameras(self):
        self.assertEqual(
            resolve_traj_filter_profile("varied_camera_3", "wrist_pick_place_no_heatmap"),
            "wrist_pick_place_no_heatmap",
        )
        self.assertEqual(
            resolve_traj_filter_profile("varied_camera_1", "wrist_pick_place_no_heatmap"),
            "external",
        )


class FindValidEpisodesTests(unittest.TestCase):
    def _write_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    def _make_valid_episode(self, base_path: Path, episode_name: str, camera_name: str = "varied_camera_1") -> Path:
        episode_dir = base_path / episode_name
        self._write_file(episode_dir / "trajectory_valid.h5")
        self._write_file(episode_dir / "rgb" / camera_name / "000000.png")
        self._write_file(episode_dir / "depth" / camera_name / "000000.npy")
        return episode_dir

    def test_accepts_prefixed_and_numeric_episode_directory_names(self):
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            episode_prefixed = self._make_valid_episode(base_path, "episode_00001")
            episode_numeric = self._make_valid_episode(base_path, "00002")
            self._make_valid_episode(base_path, "session_00003")

            episodes = find_valid_episodes(base_path, ["varied_camera_1"], "trajectory_valid.h5")

            self.assertEqual(episodes, [episode_numeric, episode_prefixed])

    def test_requires_geom_and_any_camera_with_rgb_and_depth_files(self):
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            valid_episode = self._make_valid_episode(base_path, "episode_00001", camera_name="varied_camera_2")

            missing_geom = base_path / "episode_00002"
            self._write_file(missing_geom / "rgb" / "varied_camera_1" / "000000.png")
            self._write_file(missing_geom / "depth" / "varied_camera_1" / "000000.npy")

            missing_depth = base_path / "episode_00003"
            self._write_file(missing_depth / "trajectory_valid.h5")
            self._write_file(missing_depth / "rgb" / "varied_camera_1" / "000000.png")

            wrong_suffix = base_path / "episode_00004"
            self._write_file(wrong_suffix / "trajectory_valid.h5")
            self._write_file(wrong_suffix / "rgb" / "varied_camera_1" / "000000.txt")
            self._write_file(wrong_suffix / "depth" / "varied_camera_1" / "000000.bin")

            episodes = find_valid_episodes(
                base_path,
                ["varied_camera_1", "varied_camera_2"],
                "trajectory_valid.h5",
            )

            self.assertEqual(episodes, [valid_episode])


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
        self.assertIn("--collect_profile_stats", _CLI_FLAGS)
        self.assertIn("--hardware_telemetry_interval_sec", _CLI_FLAGS)
        self.assertIn("--depth_filter_workers", _CLI_FLAGS)
        self.assertIn("--camera_num_iters", _CLI_FLAGS)
        self.assertIn("--shared_schedule_camera_names", _CLI_FLAGS)

    def test_num_iters_default_is_five(self):
        self.assertEqual(_CLI_DEFAULTS.get("--num_iters"), 5)
        self.assertIsNone(_CLI_DEFAULTS.get("--camera_num_iters"))

    def test_support_grid_ratio_default_is_point_eight(self):
        self.assertEqual(_CLI_DEFAULTS.get("--support_grid_ratio"), 0.8)

    def test_exposes_wrist_pick_place_no_heatmap_profile(self):
        self.assertIn("wrist_pick_place_no_heatmap", _CLI_CHOICES.get("--traj_filter_profile", ()))


class CameraNumItersOverrideTests(unittest.TestCase):
    def test_resolve_schedule_camera_names_defaults_to_task_cameras(self):
        resolved = resolve_schedule_camera_names(
            ["varied_camera_3"],
            None,
        )
        self.assertEqual(resolved, ["varied_camera_3"])

    def test_resolve_schedule_camera_names_accepts_explicit_override(self):
        resolved = resolve_schedule_camera_names(
            ["varied_camera_3"],
            "varied_camera_1,varied_camera_2,varied_camera_3",
        )
        self.assertEqual(
            resolved,
            ["varied_camera_1", "varied_camera_2", "varied_camera_3"],
        )

    def test_parse_camera_int_overrides_accepts_sparse_subset(self):
        overrides = parse_camera_int_overrides(
            "varied_camera_1:4,varied_camera_3:5",
            option_name="--camera_num_iters",
        )
        self.assertEqual(overrides, {"varied_camera_1": 4, "varied_camera_3": 5})

    def test_parse_camera_int_overrides_rejects_malformed_or_nonpositive_values(self):
        with self.assertRaises(ValueError):
            parse_camera_int_overrides("varied_camera_1", option_name="--camera_num_iters")
        with self.assertRaises(ValueError):
            parse_camera_int_overrides("varied_camera_1:0", option_name="--camera_num_iters")
        with self.assertRaises(ValueError):
            parse_camera_int_overrides("varied_camera_1:4,varied_camera_1:5", option_name="--camera_num_iters")

    def test_build_camera_args_applies_per_camera_num_iters_override(self):
        base_args = type(
            "Args",
            (),
            {
                "num_iters": 5,
                "camera_num_iters_overrides": {"varied_camera_1": 4},
                "traj_filter_profile": "wrist_pick_place_no_heatmap",
                "external_geom_name": "trajectory_valid.h5",
            },
        )()

        external_args = build_camera_args(
            base_args,
            Path("/tmp/episode_00001"),
            "varied_camera_1",
            query_frame_schedule_path=Path("/tmp/schedule.json"),
        )
        wrist_args = build_camera_args(
            base_args,
            Path("/tmp/episode_00001"),
            "varied_camera_3",
            query_frame_schedule_path=Path("/tmp/schedule.json"),
        )

        self.assertEqual(external_args.num_iters, 4)
        self.assertEqual(external_args.traj_filter_profile, "external")
        self.assertEqual(wrist_args.num_iters, 5)
        self.assertEqual(wrist_args.traj_filter_profile, "wrist_pick_place_no_heatmap")


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
                "camera_num_iters_overrides": {"varied_camera_1": 4},
                "depth_filter_workers": 12,
                "traj_filter_profile": "external",
            },
        )()

        record = build_camera_task_metric_record(
            task=task,
            gpu_id=0,
            args=args,
            worker_label="GPU 0 slot 1/4",
            worker_index=1,
            gpu_slot_index=1,
            gpu_slot_count=4,
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
        self.assertEqual(record["camera_num_iters_overrides"], {"varied_camera_1": 4})
        self.assertEqual(record["depth_filter_workers"], 12)
        self.assertEqual(record["worker_label"], "GPU 0 slot 1/4")
        self.assertEqual(record["query_frame_count"], 10)
        self.assertAlmostEqual(record["process_seconds_per_query"], 5.0)
        self.assertAlmostEqual(record["save_seconds_per_query"], 0.2)
        self.assertAlmostEqual(record["total_seconds_per_query"], 5.2)
        self.assertEqual(record["status"], "success")

    def test_build_camera_task_profile_record_keeps_nested_profile_stats(self):
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
                "camera_num_iters_overrides": {"varied_camera_1": 4},
                "depth_filter_workers": 8,
                "traj_filter_profile": "external",
            },
        )()

        record = build_camera_task_profile_record(
            task=task,
            gpu_id=0,
            args=args,
            worker_label="GPU 0 slot 1/2",
            worker_index=1,
            gpu_slot_index=1,
            gpu_slot_count=2,
            query_frame_count=3,
            process_seconds=12.0,
            save_seconds=1.5,
            started_at_unix=100.0,
            finished_at_unix=113.5,
            status="success",
            retryable_cuda_error=False,
            error_message=None,
            profile_stats={"tracker_model_forward_seconds": 10.0},
            save_profile_stats={"sample_write_seconds": 1.0},
            per_query_save_seconds={7: 0.5, 9: 0.7},
            scene_finalize_overhead_seconds=0.3,
        )

        self.assertEqual(record["worker_label"], "GPU 0 slot 1/2")
        self.assertEqual(record["query_frame_count"], 3)
        self.assertEqual(record["camera_num_iters_overrides"], {"varied_camera_1": 4})
        self.assertEqual(record["profile_stats"]["tracker_model_forward_seconds"], 10.0)
        self.assertEqual(record["save_profile_stats"]["sample_write_seconds"], 1.0)
        self.assertEqual(record["per_query_save_seconds"], {"7": 0.5, "9": 0.7})
        self.assertAlmostEqual(record["scene_finalize_overhead_seconds"], 0.3)

    def test_build_batch_run_summary_preserves_fixed_keyframe_config(self):
        args = type(
            "Args",
            (),
            {
                "camera_names": ["varied_camera_1", "varied_camera_3"],
                "shared_schedule_camera_names": ["varied_camera_1", "varied_camera_2", "varied_camera_3"],
                "workers_per_gpu": 4,
                "collect_profile_stats": True,
                "hardware_telemetry_interval_sec": 30.0,
                "num_iters": 6,
                "camera_num_iters_overrides": {"varied_camera_1": 4, "varied_camera_2": 4},
                "depth_filter_workers": 16,
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
            telemetry_gpu_ids=[0, 1, 2, 3],
            host_name="worker-host",
            gpu_info=[{"gpu_id": 0, "name": "NVIDIA H200"}],
            worker_slot_count=16,
            episode_count=30,
            camera_task_count=60,
            total_camera_success=60,
            total_camera_fail=0,
            wall_clock_seconds=1234.5,
        )

        self.assertEqual(summary["camera_names"], ["varied_camera_1", "varied_camera_3"])
        self.assertEqual(
            summary["shared_schedule_camera_names"],
            ["varied_camera_1", "varied_camera_2", "varied_camera_3"],
        )
        self.assertEqual(summary["gpu_ids"], [0, 1, 2, 3])
        self.assertEqual(summary["telemetry_gpu_ids"], [0, 1, 2, 3])
        self.assertEqual(summary["host_name"], "worker-host")
        self.assertEqual(summary["workers_per_gpu"], 4)
        self.assertEqual(summary["worker_slot_count"], 16)
        self.assertTrue(summary["collect_profile_stats"])
        self.assertAlmostEqual(summary["hardware_telemetry_interval_sec"], 30.0)
        self.assertEqual(summary["camera_num_iters_overrides"], {"varied_camera_1": 4, "varied_camera_2": 4})
        self.assertEqual(summary["depth_filter_workers"], 16)
        self.assertEqual(summary["episode_count"], 30)
        self.assertEqual(summary["camera_task_count"], 60)
        self.assertEqual(summary["keyframes_per_sec_min"], 5)
        self.assertEqual(summary["keyframes_per_sec_max"], 5)
        self.assertAlmostEqual(summary["support_grid_ratio"], 0.8)
        self.assertAlmostEqual(summary["wall_clock_seconds"], 1234.5)


if __name__ == "__main__":
    unittest.main()
