import ast
import os
import unittest
from pathlib import Path


_SOURCE_PATH = Path(__file__).resolve().with_name("infer.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_LOAD_VIDEO_AND_MASK_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "load_video_and_mask"
)
_MODULE = ast.Module(body=[_LOAD_VIDEO_AND_MASK_AST], type_ignores=[])
_GLOBALS = {
    "os": os,
}
exec(compile(_MODULE, str(_SOURCE_PATH), "exec"), _GLOBALS)
load_video_and_mask = _GLOBALS["load_video_and_mask"]


class LoadVideoAndMaskTests(unittest.TestCase):
    def test_missing_path_raises_file_not_found(self):
        missing_path = "/tmp/traceforge_missing_video_dir"

        with self.assertRaises(FileNotFoundError) as exc_info:
            load_video_and_mask(missing_path, None, 1, 512, False)

        self.assertIn(missing_path, str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
