import numpy as np


def invert_extrinsics_batch(extrinsics: np.ndarray, *, context: str) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(
            f"{context}: expected extrinsics with shape (T, 4, 4), got {extrinsics.shape}"
        )

    try:
        return np.linalg.inv(extrinsics).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{context}: failed to invert extrinsics; all matrices must be valid 4x4 transforms."
        ) from exc


def normalize_extrinsics_to_w2c(
    extrinsics: np.ndarray,
    *,
    extr_mode: str,
    context: str,
) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(
            f"{context}: expected extrinsics with shape (T, 4, 4), got {extrinsics.shape}"
        )

    if extr_mode == "w2c":
        return extrinsics.copy()
    if extr_mode == "c2w":
        return invert_extrinsics_batch(
            extrinsics,
            context=f"{context} (convert c2w -> w2c)",
        )

    raise ValueError(
        f"{context}: unknown extr_mode='{extr_mode}', expected 'w2c' or 'c2w'."
    )
