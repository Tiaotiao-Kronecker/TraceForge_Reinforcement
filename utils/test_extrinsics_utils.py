import unittest

import numpy as np

from utils.extrinsics_utils import invert_extrinsics_batch, normalize_extrinsics_to_w2c


def _translation(tx: float, ty: float, tz: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return mat


class ExtrinsicsUtilsTests(unittest.TestCase):
    def test_invert_extrinsics_batch_inverts_translation(self):
        c2w = _translation(1.0, -2.0, 3.5)[None]

        w2c = invert_extrinsics_batch(c2w, context="test invert")

        np.testing.assert_allclose(w2c[0], _translation(-1.0, 2.0, -3.5), atol=1e-6)

    def test_normalize_extrinsics_to_w2c_keeps_w2c_input(self):
        w2c = _translation(-0.25, 0.5, -1.0)[None]

        out = normalize_extrinsics_to_w2c(
            w2c,
            extr_mode="w2c",
            context="test normalize",
        )

        np.testing.assert_allclose(out, w2c, atol=1e-6)

    def test_normalize_extrinsics_to_w2c_inverts_c2w_input(self):
        c2w = _translation(0.25, -0.5, 1.0)[None]

        out = normalize_extrinsics_to_w2c(
            c2w,
            extr_mode="c2w",
            context="test normalize",
        )

        np.testing.assert_allclose(out[0], _translation(-0.25, 0.5, -1.0), atol=1e-6)

    def test_normalize_extrinsics_to_w2c_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "unknown extr_mode"):
            normalize_extrinsics_to_w2c(
                np.eye(4, dtype=np.float32)[None],
                extr_mode="bad_mode",
                context="test normalize",
            )


if __name__ == "__main__":
    unittest.main()
