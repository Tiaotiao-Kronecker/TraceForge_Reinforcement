from __future__ import annotations

import math

import numpy as np


def build_candidate_source_frame_indices(
    num_frames: int,
    *,
    stride: int,
    max_num_frames: int | None,
) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.int32)

    indices = np.arange(num_frames, dtype=np.int32)[::stride]
    if max_num_frames is not None and max_num_frames > 0:
        indices = indices[:max_num_frames]
    return indices.astype(np.int32, copy=False)


def _sample_stratified_without_replacement(
    source_indices: np.ndarray,
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source_indices = np.asarray(source_indices, dtype=np.int32).reshape(-1)
    if num_samples <= 0 or source_indices.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if num_samples >= source_indices.size:
        return np.sort(source_indices.astype(np.int32, copy=True))

    boundaries = np.linspace(0, source_indices.size, num_samples + 1, dtype=np.int32)
    selected: list[int] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        offset = int(rng.integers(start, end))
        selected.append(int(source_indices[offset]))
    if not selected:
        return np.zeros((0,), dtype=np.int32)
    return np.sort(np.asarray(selected, dtype=np.int32))


def sample_query_source_indices_per_second(
    candidate_source_indices: np.ndarray,
    *,
    episode_fps: float,
    keyframes_per_sec_min: int,
    keyframes_per_sec_max: int,
    seed: int,
) -> np.ndarray:
    if episode_fps <= 0:
        raise ValueError(f"episode_fps must be positive, got {episode_fps}")
    if keyframes_per_sec_min <= 0 or keyframes_per_sec_max <= 0:
        raise ValueError(
            "keyframes_per_sec_min and keyframes_per_sec_max must both be positive"
        )
    if keyframes_per_sec_min > keyframes_per_sec_max:
        raise ValueError(
            f"keyframes_per_sec_min ({keyframes_per_sec_min}) must be <= "
            f"keyframes_per_sec_max ({keyframes_per_sec_max})"
        )

    candidate_source_indices = np.asarray(candidate_source_indices, dtype=np.int32).reshape(-1)
    if candidate_source_indices.size == 0:
        return np.zeros((0,), dtype=np.int32)

    rng = np.random.default_rng(seed)
    duration_sec = int(math.floor(float(candidate_source_indices[-1]) / float(episode_fps))) + 1
    selected_chunks: list[np.ndarray] = []

    for second_idx in range(duration_sec):
        sec_start = float(second_idx)
        sec_end = float(second_idx + 1)
        mask = (
            (candidate_source_indices.astype(np.float64) / float(episode_fps)) >= sec_start
        ) & (
            (candidate_source_indices.astype(np.float64) / float(episode_fps)) < sec_end
        )
        second_candidates = candidate_source_indices[mask]
        if second_candidates.size == 0:
            continue

        if keyframes_per_sec_min == keyframes_per_sec_max:
            num_samples = keyframes_per_sec_min
        else:
            num_samples = int(
                rng.integers(keyframes_per_sec_min, keyframes_per_sec_max + 1)
            )
        num_samples = min(num_samples, int(second_candidates.size))
        selected_chunks.append(
            _sample_stratified_without_replacement(
                second_candidates,
                num_samples=num_samples,
                rng=rng,
            )
        )

    if not selected_chunks:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(selected_chunks, axis=0).astype(np.int32, copy=False)


def map_query_source_indices_to_local(
    source_frame_indices: np.ndarray,
    query_source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.int32).reshape(-1)
    query_source_indices = np.asarray(query_source_indices, dtype=np.int32).reshape(-1)

    source_to_local = {
        int(source_idx): local_idx
        for local_idx, source_idx in enumerate(source_frame_indices.tolist())
    }

    local_indices: list[int] = []
    missing_source_indices: list[int] = []
    seen_local: set[int] = set()
    for source_idx in query_source_indices.tolist():
        local_idx = source_to_local.get(int(source_idx))
        if local_idx is None:
            missing_source_indices.append(int(source_idx))
            continue
        if local_idx in seen_local:
            continue
        seen_local.add(local_idx)
        local_indices.append(int(local_idx))

    return (
        np.asarray(local_indices, dtype=np.int32),
        np.asarray(missing_source_indices, dtype=np.int32),
    )
