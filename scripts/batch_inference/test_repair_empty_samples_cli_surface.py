import ast
import unittest
from pathlib import Path


_SOURCE_PATH = Path(__file__).resolve().with_name("repair_empty_samples_press_one_button_demo.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_PARSE_ARGS_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
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


class RepairEmptySamplesCliSurfaceTests(unittest.TestCase):
    def test_camera_names_has_no_hardcoded_default(self):
        self.assertIsNone(_collect_cli_default(_PARSE_ARGS_AST, "--camera_names"))

    def test_num_iters_default_is_three(self):
        self.assertEqual(_collect_cli_default(_PARSE_ARGS_AST, "--num_iters"), 3)


if __name__ == "__main__":
    unittest.main()
