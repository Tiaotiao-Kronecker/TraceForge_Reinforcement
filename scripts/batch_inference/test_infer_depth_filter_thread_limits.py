import ast
import unittest
from pathlib import Path


_SOURCE_PATH = Path(__file__).resolve().with_name("infer.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_APPLY_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "_apply_native_thread_limiters"
)
_RESTORE_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "_restore_native_thread_limiters"
)
_MODULE = ast.Module(body=[_APPLY_AST, _RESTORE_AST], type_ignores=[])
_GLOBALS: dict[str, object] = {}
exec(compile(_MODULE, str(_SOURCE_PATH), "exec"), _GLOBALS)
apply_native_thread_limiters = _GLOBALS["_apply_native_thread_limiters"]
restore_native_thread_limiters = _GLOBALS["_restore_native_thread_limiters"]


class _FakeThreadLimiter:
    def __init__(self, initial_threads: int) -> None:
        self.threads = int(initial_threads)
        self.calls: list[int] = []

    def get_threads(self) -> int:
        return self.threads

    def set_threads(self, value: int) -> None:
        self.calls.append(int(value))
        self.threads = int(value)


class DepthFilterThreadLimitHelperTests(unittest.TestCase):
    def test_apply_and_restore_round_trip(self):
        openblas = _FakeThreadLimiter(64)
        scipy_openblas = _FakeThreadLimiter(8)

        previous_limits = apply_native_thread_limiters(
            [
                (openblas.get_threads, openblas.set_threads),
                (scipy_openblas.get_threads, scipy_openblas.set_threads),
            ],
            1,
        )

        self.assertEqual(openblas.calls, [1])
        self.assertEqual(scipy_openblas.calls, [1])
        self.assertEqual(openblas.threads, 1)
        self.assertEqual(scipy_openblas.threads, 1)

        restore_native_thread_limiters(previous_limits)

        self.assertEqual(openblas.calls, [1, 64])
        self.assertEqual(scipy_openblas.calls, [1, 8])
        self.assertEqual(openblas.threads, 64)
        self.assertEqual(scipy_openblas.threads, 8)

    def test_apply_skips_redundant_set_when_already_at_target(self):
        limiter = _FakeThreadLimiter(1)

        previous_limits = apply_native_thread_limiters(
            [(limiter.get_threads, limiter.set_threads)],
            1,
        )

        self.assertEqual(limiter.calls, [])

        restore_native_thread_limiters(previous_limits)

        self.assertEqual(limiter.calls, [])


if __name__ == "__main__":
    unittest.main()
