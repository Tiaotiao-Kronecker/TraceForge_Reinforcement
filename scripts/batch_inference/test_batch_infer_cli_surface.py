import ast
import unittest
from pathlib import Path


_SOURCE_PATH = Path(__file__).resolve().with_name("batch_infer.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_XPERIENCE_SOURCE_PATH = Path(__file__).resolve().parent / "adapters" / "xperience_raw.py"
_XPERIENCE_SOURCE_AST = ast.parse(_XPERIENCE_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_XPERIENCE_SOURCE_PATH))


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


def _collect_argument_calls(func_ast: ast.FunctionDef) -> dict[str, ast.Call]:
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                calls[arg.value] = node
                break
    return calls


def _get_keyword_value(call_ast: ast.Call, keyword_name: str):
    for keyword in call_ast.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def _collect_constant_tuple(module_ast: ast.Module, name: str) -> tuple[str, ...]:
    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        if not isinstance(node.value, ast.Tuple):
            raise AssertionError(f"{name} must be assigned from a tuple literal")
        values: list[str] = []
        for element in node.value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                raise AssertionError(f"{name} must contain only string literals")
            values.append(element.value)
        return tuple(values)
    raise AssertionError(f"missing assignment for {name}")


_BUILD_DISPATCH_PARSER_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "_build_dispatch_parser"
)
_BUILD_XPERIENCE_PARSER_AST = next(
    node for node in _XPERIENCE_SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "build_parser"
)

_DISPATCH_FLAGS = _collect_cli_flags(_BUILD_DISPATCH_PARSER_AST)
_SUPPORTED_DATASET_ADAPTERS = _collect_constant_tuple(_SOURCE_AST, "SUPPORTED_DATASET_ADAPTERS")
_XPERIENCE_FLAGS = _collect_cli_flags(_BUILD_XPERIENCE_PARSER_AST)
_XPERIENCE_CHOICES = _collect_cli_choices(_BUILD_XPERIENCE_PARSER_AST)
_XPERIENCE_ARGUMENT_CALLS = _collect_argument_calls(_BUILD_XPERIENCE_PARSER_AST)


class BatchInferCliSurfaceTests(unittest.TestCase):
    def test_dispatch_parser_exposes_dataset_adapter_flag(self):
        self.assertIn("--dataset_adapter", _DISPATCH_FLAGS)
        self.assertEqual(_SUPPORTED_DATASET_ADAPTERS, ("sim_file_layout", "xperience_raw"))

    def test_xperience_parser_exposes_native_dataset_controls(self):
        self.assertIn("--dataset_root", _XPERIENCE_FLAGS)
        self.assertIn("--episode_glob", _XPERIENCE_FLAGS)
        self.assertIn("--window_size", _XPERIENCE_FLAGS)
        self.assertIn("--window_step", _XPERIENCE_FLAGS)
        self.assertIn("--camera_name", _XPERIENCE_FLAGS)
        self.assertIn("--gpu_id", _XPERIENCE_FLAGS)
        self.assertIn("--workers_per_gpu", _XPERIENCE_FLAGS)
        self.assertIn("--min_free_gpu_mem_gb", _XPERIENCE_FLAGS)
        self.assertIn("--gpu_recovery_poll_sec", _XPERIENCE_FLAGS)
        self.assertIn("--telemetry_out_dir", _XPERIENCE_FLAGS)
        self.assertIn("--hardware_telemetry_interval_sec", _XPERIENCE_FLAGS)
        self.assertIn("--keyframes_per_sec_min", _XPERIENCE_FLAGS)
        self.assertIn("--keyframes_per_sec_max", _XPERIENCE_FLAGS)
        self.assertIn("--keyframe_seed", _XPERIENCE_FLAGS)
        self.assertIn("--fallback_episode_fps", _XPERIENCE_FLAGS)
        self.assertIn("--scene_storage_mode", _XPERIENCE_FLAGS)
        self.assertIn("--tracker_precision_mode", _XPERIENCE_FLAGS)
        self.assertIn("--query_visibility_gate_mode", _XPERIENCE_FLAGS)
        self.assertIn("--query_depth_stabilization_mode", _XPERIENCE_FLAGS)
        self.assertIn("--grid_width", _XPERIENCE_FLAGS)
        self.assertIn("--grid_height", _XPERIENCE_FLAGS)

    def test_xperience_parser_defaults_to_stereo_left_and_adapter_ref(self):
        camera_name_call = _XPERIENCE_ARGUMENT_CALLS["--camera_name"]
        camera_default = _get_keyword_value(camera_name_call, "default")
        self.assertIsInstance(camera_default, ast.Constant)
        self.assertEqual(camera_default.value, "stereo_left")

        scene_storage_call = _XPERIENCE_ARGUMENT_CALLS["--scene_storage_mode"]
        scene_storage_default = _get_keyword_value(scene_storage_call, "default")
        self.assertIsInstance(scene_storage_default, ast.Name)
        self.assertEqual(scene_storage_default.id, "SCENE_STORAGE_ADAPTER_REF")

        scene_storage_choices = _get_keyword_value(scene_storage_call, "choices")
        self.assertIsInstance(scene_storage_choices, ast.List)
        self.assertEqual(
            [element.id for element in scene_storage_choices.elts if isinstance(element, ast.Name)],
            ["SCENE_STORAGE_ADAPTER_REF", "SCENE_STORAGE_CACHE"],
        )

    def test_xperience_parser_defaults_to_per_second_keyframe_sampling(self):
        min_call = _XPERIENCE_ARGUMENT_CALLS["--keyframes_per_sec_min"]
        min_default = _get_keyword_value(min_call, "default")
        self.assertIsInstance(min_default, ast.Constant)
        self.assertEqual(min_default.value, 2)

        max_call = _XPERIENCE_ARGUMENT_CALLS["--keyframes_per_sec_max"]
        max_default = _get_keyword_value(max_call, "default")
        self.assertIsInstance(max_default, ast.Constant)
        self.assertEqual(max_default.value, 3)

    def test_xperience_parser_defaults_to_future_len_16(self):
        future_len_call = _XPERIENCE_ARGUMENT_CALLS["--future_len"]
        future_len_default = _get_keyword_value(future_len_call, "default")
        self.assertIsInstance(future_len_default, ast.Constant)
        self.assertEqual(future_len_default.value, 16)

    def test_xperience_parser_defaults_to_one_worker_per_gpu(self):
        workers_call = _XPERIENCE_ARGUMENT_CALLS["--workers_per_gpu"]
        workers_default = _get_keyword_value(workers_call, "default")
        self.assertIsInstance(workers_default, ast.Constant)
        self.assertEqual(workers_default.value, 1)


if __name__ == "__main__":
    unittest.main()
