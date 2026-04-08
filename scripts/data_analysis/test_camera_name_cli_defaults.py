import ast
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_DEPTH_VOL_PATH = _ROOT / "benchmark_depth_volatility_optimization.py"
_DEPTH_VOL_AST = ast.parse(_DEPTH_VOL_PATH.read_text(encoding="utf-8"), filename=str(_DEPTH_VOL_PATH))
_DEPTH_VOL_PARSE_ARGS_AST = next(
    node for node in _DEPTH_VOL_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
)
_DEPTH_VOL_PARSE_CAMERA_NAMES_AST = next(
    node for node in _DEPTH_VOL_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_camera_names"
)
_DEPTH_VOL_MODULE = ast.Module(body=[_DEPTH_VOL_PARSE_CAMERA_NAMES_AST], type_ignores=[])
_DEPTH_VOL_GLOBALS: dict[str, object] = {}
exec(compile(_DEPTH_VOL_MODULE, str(_DEPTH_VOL_PATH), "exec"), _DEPTH_VOL_GLOBALS)
parse_camera_names = _DEPTH_VOL_GLOBALS["parse_camera_names"]


def _load_parse_args_ast(path: Path) -> ast.FunctionDef:
    source_ast = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node for node in source_ast.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )


def _collect_cli_default(func_ast: ast.FunctionDef, flag: str) -> object:
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        has_flag = any(
            isinstance(arg, ast.Constant) and arg.value == flag
            for arg in node.args
        )
        if not has_flag:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
        return None
    raise AssertionError(f"missing CLI flag {flag} in {func_ast.name}")


class CameraNameParsingTests(unittest.TestCase):
    def test_parse_camera_names_requires_explicit_value(self):
        with self.assertRaises(ValueError):
            parse_camera_names(None)
        with self.assertRaises(ValueError):
            parse_camera_names("")


class CameraNameCliDefaultTests(unittest.TestCase):
    def test_benchmark_depth_volatility_requires_explicit_camera_names(self):
        self.assertIsNone(_collect_cli_default(_DEPTH_VOL_PARSE_ARGS_AST, "--camera-names"))

    def test_benchmark_inference_variants_requires_explicit_camera_names(self):
        parse_args_ast = _load_parse_args_ast(_ROOT / "benchmark_inference_variants.py")
        self.assertIsNone(_collect_cli_default(parse_args_ast, "--camera-names"))

    def test_benchmark_num_iters_sweep_requires_explicit_camera_names(self):
        parse_args_ast = _load_parse_args_ast(_ROOT / "benchmark_num_iters_sweep.py")
        self.assertIsNone(_collect_cli_default(parse_args_ast, "--camera-names"))

    def test_benchmark_num_iters_manifest_requires_explicit_camera_names(self):
        parse_args_ast = _load_parse_args_ast(_ROOT / "benchmark_num_iters_manifest.py")
        self.assertIsNone(_collect_cli_default(parse_args_ast, "--camera-names"))

    def test_compare_traceforge_output_roots_requires_explicit_camera_names(self):
        parse_args_ast = _load_parse_args_ast(_ROOT / "compare_traceforge_output_roots.py")
        self.assertIsNone(_collect_cli_default(parse_args_ast, "--camera-names"))


if __name__ == "__main__":
    unittest.main()
