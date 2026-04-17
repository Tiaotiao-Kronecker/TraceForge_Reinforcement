from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


H100_PEAK_TFLOPS = {
    # Dense throughput figures. BF16 values are derived from NVIDIA's published
    # sparsity-on numbers by dividing by two.
    "sxm": {"fp32": 67.0, "bf16": 989.5},
    "nvl": {"fp32": 60.0, "bf16": 835.0},
}


@dataclass(frozen=True)
class MinimalTrackerCase:
    name: str
    video: np.ndarray
    depths: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    query_points_world: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedTrackerCase:
    name: str
    video: torch.Tensor
    depths: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    query_points_world: torch.Tensor


@dataclass(frozen=True)
class SymbolicComplexityReport:
    num_frames: int
    frame_height: int
    frame_width: int
    frame_pixels: int
    seq_len: int
    window_count: int
    query_count: int
    support_query_count: int
    total_query_count: int
    num_iters: int
    shared_unprojection_points: int
    repeated_window_unprojection_points: int
    total_unprojection_points: int
    iterative_track_state_updates: int
    iterative_query_frame_updates: int


@dataclass(frozen=True)
class RooflineEstimate:
    h100_variant: str
    precision_mode: str
    peak_tflops: float
    profiled_flops: int
    theoretical_min_seconds: float | None


@dataclass(frozen=True)
class MinimalTrackerRunResult:
    precision_mode: str
    wall_time_seconds: float
    profile_stats: dict[str, float]
    support_query_count: int
    output_shape: tuple[int, ...]
    visibility_shape: tuple[int, ...]
    profiled_flops: int | None = None
    top_profiler_ops: list[dict[str, Any]] | None = None
    effective_tflops: float | None = None
    roofline: RooflineEstimate | None = None


def load_case_from_npz(npz_path: str | Path) -> MinimalTrackerCase:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        video = np.asarray(data["video"])
        depths = np.asarray(data["depths"])
        intrinsics = np.asarray(data["intrinsics"])
        extrinsics = np.asarray(data["extrinsics"])
        query_points_world = None
        if "query_points_world" in data:
            query_points_world = np.asarray(data["query_points_world"])
        name = str(data["name"]) if "name" in data else npz_path.stem
    return MinimalTrackerCase(
        name=name,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        query_points_world=query_points_world,
    )


def create_synthetic_case(
    *,
    frames: int,
    height: int,
    width: int,
    depth_m: float = 1.0,
    name: str = "synthetic",
) -> MinimalTrackerCase:
    if frames <= 0:
        raise ValueError("frames must be >= 1")
    if height <= 1 or width <= 1:
        raise ValueError("height and width must be > 1")
    video = np.linspace(
        0.0,
        1.0,
        num=frames * height * width * 3,
        dtype=np.float32,
    ).reshape(frames, height, width, 3)
    depths = np.full((frames, height, width), float(depth_m), dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0)
    intrinsics[:, 0, 0] = max(width, height)
    intrinsics[:, 1, 1] = max(width, height)
    intrinsics[:, 0, 2] = (width - 1) * 0.5
    intrinsics[:, 1, 2] = (height - 1) * 0.5
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    return MinimalTrackerCase(
        name=name,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        query_points_world=None,
    )


def build_world_grid_queries(
    *,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    grid_size: int,
    query_frame: int = 0,
) -> np.ndarray:
    if grid_size <= 0:
        raise ValueError("grid_size must be >= 1")
    depths = np.asarray(depths, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if depths.ndim != 3:
        raise ValueError(f"depths must have shape [T, H, W], got {depths.shape}")
    if intrinsics.shape != (depths.shape[0], 3, 3):
        raise ValueError(f"intrinsics must have shape [T, 3, 3], got {intrinsics.shape}")
    if extrinsics.shape != (depths.shape[0], 4, 4):
        raise ValueError(f"extrinsics must have shape [T, 4, 4], got {extrinsics.shape}")
    if query_frame < 0 or query_frame >= depths.shape[0]:
        raise ValueError(f"query_frame {query_frame} is out of range for T={depths.shape[0]}")

    frame_depth = depths[query_frame]
    height, width = frame_depth.shape
    x_coords = np.linspace(0.0, float(width - 1), grid_size, dtype=np.float32)
    y_coords = np.linspace(0.0, float(height - 1), grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    ji = np.round(xy).astype(np.int32, copy=False)

    sampled_depth = frame_depth[ji[:, 1], ji[:, 0]]
    valid_mask = sampled_depth > 0.0
    if not np.any(valid_mask):
        raise ValueError("grid query construction found no valid depth values")

    xy = xy[valid_mask]
    sampled_depth = sampled_depth[valid_mask]
    intrinsic = intrinsics[query_frame]
    extrinsic = extrinsics[query_frame]

    k_inv = np.linalg.inv(intrinsic)
    c2w = np.linalg.inv(extrinsic)
    xy_homo = np.concatenate(
        [xy, np.ones((xy.shape[0], 1), dtype=np.float32)],
        axis=-1,
    )
    local_coords = (k_inv @ xy_homo.T).T * sampled_depth[:, None]
    world_coords = (c2w[:3, :3] @ local_coords.T).T + c2w[:3, 3]
    time_column = np.full((world_coords.shape[0], 1), float(query_frame), dtype=np.float32)
    return np.concatenate([time_column, world_coords.astype(np.float32, copy=False)], axis=-1)


def prepare_tracker_case(
    case: MinimalTrackerCase,
    *,
    device: str | torch.device,
    query_grid_size: int,
    query_frame: int,
) -> PreparedTrackerCase:
    device = torch.device(device)
    video = np.asarray(case.video)
    if video.ndim != 4:
        raise ValueError(f"video must have shape [T, H, W, 3] or [T, 3, H, W], got {video.shape}")
    if video.shape[-1] == 3:
        video = np.transpose(video, (0, 3, 1, 2))
    elif video.shape[1] != 3:
        raise ValueError(f"video channel dimension is not recognized: {video.shape}")
    video = video.astype(np.float32, copy=False)
    if video.max() > 1.0:
        video = video / 255.0

    depths = np.asarray(case.depths, dtype=np.float32)
    intrinsics = np.asarray(case.intrinsics, dtype=np.float32)
    extrinsics = np.asarray(case.extrinsics, dtype=np.float32)
    if case.query_points_world is None:
        query_points_world = build_world_grid_queries(
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            grid_size=query_grid_size,
            query_frame=query_frame,
        )
    else:
        query_points_world = np.asarray(case.query_points_world, dtype=np.float32)
        if query_points_world.ndim != 2 or query_points_world.shape[1] != 4:
            raise ValueError(
                "query_points_world must have shape [N, 4] with columns [t, x, y, z]"
            )

    return PreparedTrackerCase(
        name=case.name,
        video=torch.from_numpy(video).to(device=device, dtype=torch.float32),
        depths=torch.from_numpy(depths).to(device=device, dtype=torch.float32),
        intrinsics=torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32),
        extrinsics=torch.from_numpy(extrinsics).to(device=device, dtype=torch.float32),
        query_points_world=torch.from_numpy(query_points_world).to(device=device, dtype=torch.float32),
    )


def load_tracker_model(checkpoint_path: str | Path, device: str | torch.device) -> torch.nn.Module:
    from utils.inference_utils import load_model

    model = load_model(str(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model


def estimate_streaming_window_count(num_frames: int, seq_len: int) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be >= 1")
    if seq_len <= 0 or seq_len % 2 != 0:
        raise ValueError("seq_len must be a positive even integer")
    half = seq_len // 2
    padded_frames = num_frames
    if padded_frames % half != 0:
        padded_frames += half - (padded_frames % half)
    padded_frames = max(padded_frames, seq_len)
    return len(range(seq_len, padded_frames + 1, half))


def build_symbolic_complexity_report(
    *,
    prepared_case: PreparedTrackerCase,
    num_iters: int,
    seq_len: int,
    support_query_count: int,
) -> SymbolicComplexityReport:
    num_frames, _, height, width = prepared_case.video.shape
    frame_pixels = int(height * width)
    query_count = int(prepared_case.query_points_world.shape[0])
    total_query_count = int(query_count + support_query_count)
    window_count = estimate_streaming_window_count(num_frames=num_frames, seq_len=seq_len)
    return SymbolicComplexityReport(
        num_frames=int(num_frames),
        frame_height=int(height),
        frame_width=int(width),
        frame_pixels=frame_pixels,
        seq_len=int(seq_len),
        window_count=int(window_count),
        query_count=query_count,
        support_query_count=int(support_query_count),
        total_query_count=total_query_count,
        num_iters=int(num_iters),
        shared_unprojection_points=int(num_frames * frame_pixels),
        repeated_window_unprojection_points=int(window_count * num_frames * frame_pixels),
        total_unprojection_points=int((window_count + 1) * num_frames * frame_pixels),
        iterative_track_state_updates=int(window_count * num_iters * seq_len * total_query_count),
        iterative_query_frame_updates=int(window_count * seq_len * total_query_count),
    )


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _get_autocast_context(device: torch.device, precision_mode: str):
    from utils.inference_utils import get_tracker_precision_autocast_context

    return get_tracker_precision_autocast_context(device, precision_mode)


@torch.inference_mode()
def run_tracker_once(
    *,
    model: torch.nn.Module,
    prepared_case: PreparedTrackerCase,
    num_iters: int,
    support_grid_size: int,
    precision_mode: str,
) -> MinimalTrackerRunResult:
    from utils.inference_utils import configure_tracker_precision, inference, normalize_tracker_precision_mode

    device = prepared_case.video.device
    profile_stats: dict[str, float] = {}
    precision_mode = normalize_tracker_precision_mode(precision_mode)
    configure_tracker_precision(model, precision_mode)
    autocast_ctx = _get_autocast_context(device, precision_mode)
    _sync_if_cuda(device)
    started = time.perf_counter()
    with autocast_ctx:
        coords, visibs, metadata = inference(
            model=model,
            video=prepared_case.video,
            depths=prepared_case.depths,
            intrinsics=prepared_case.intrinsics,
            extrinsics=prepared_case.extrinsics,
            query_point=prepared_case.query_points_world,
            num_iters=num_iters,
            grid_size=support_grid_size,
            bidrectional=False,
            tracker_precision_mode=precision_mode,
            profile_stats=profile_stats,
            return_metadata=True,
        )
    _sync_if_cuda(device)
    wall_time_seconds = time.perf_counter() - started
    return MinimalTrackerRunResult(
        precision_mode=precision_mode,
        wall_time_seconds=float(wall_time_seconds),
        profile_stats=profile_stats,
        support_query_count=int(metadata.get("effective_support_query_count", 0)),
        output_shape=tuple(int(v) for v in coords.shape),
        visibility_shape=tuple(int(v) for v in visibs.shape),
    )


def profile_tracker_flops(
    *,
    model: torch.nn.Module,
    prepared_case: PreparedTrackerCase,
    num_iters: int,
    support_grid_size: int,
    precision_mode: str,
    top_k: int = 20,
) -> tuple[int, list[dict[str, Any]]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if prepared_case.video.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        run_tracker_once(
            model=model,
            prepared_case=prepared_case,
            num_iters=num_iters,
            support_grid_size=support_grid_size,
            precision_mode=precision_mode,
        )
    events: list[dict[str, Any]] = []
    total_flops = 0
    for event in prof.key_averages():
        flops = int(getattr(event, "flops", 0) or 0)
        total_flops += flops
        if flops <= 0:
            continue
        events.append(
            {
                "op": event.key,
                "flops": flops,
                "cpu_time_ms": float(event.self_cpu_time_total / 1000.0),
            }
        )
    events.sort(key=lambda item: item["flops"], reverse=True)
    return total_flops, events[:top_k]


def build_roofline_estimate(
    *,
    profiled_flops: int,
    h100_variant: str,
    precision_mode: str,
) -> RooflineEstimate:
    roofline_precision_mode = str(precision_mode)
    if roofline_precision_mode in {"autocast_bf16", "deep_bf16"}:
        roofline_precision_mode = "bf16"
    if h100_variant not in H100_PEAK_TFLOPS:
        raise ValueError(f"Unsupported h100_variant: {h100_variant}")
    if roofline_precision_mode not in H100_PEAK_TFLOPS[h100_variant]:
        raise ValueError(f"Unsupported precision_mode: {precision_mode}")
    peak_tflops = float(H100_PEAK_TFLOPS[h100_variant][roofline_precision_mode])
    theoretical_min_seconds = None
    if profiled_flops > 0:
        theoretical_min_seconds = float(profiled_flops / (peak_tflops * 1e12))
    return RooflineEstimate(
        h100_variant=h100_variant,
        precision_mode=precision_mode,
        peak_tflops=peak_tflops,
        profiled_flops=int(profiled_flops),
        theoretical_min_seconds=theoretical_min_seconds,
    )


def attach_profiler_summary(
    run_result: MinimalTrackerRunResult,
    *,
    profiled_flops: int,
    top_profiler_ops: list[dict[str, Any]],
    h100_variant: str,
) -> MinimalTrackerRunResult:
    roofline = build_roofline_estimate(
        profiled_flops=profiled_flops,
        h100_variant=h100_variant,
        precision_mode=run_result.precision_mode,
    )
    effective_tflops = None
    if run_result.wall_time_seconds > 0.0 and profiled_flops > 0:
        effective_tflops = float(profiled_flops / run_result.wall_time_seconds / 1e12)
    return MinimalTrackerRunResult(
        precision_mode=run_result.precision_mode,
        wall_time_seconds=run_result.wall_time_seconds,
        profile_stats=dict(run_result.profile_stats),
        support_query_count=run_result.support_query_count,
        output_shape=run_result.output_shape,
        visibility_shape=run_result.visibility_shape,
        profiled_flops=int(profiled_flops),
        top_profiler_ops=top_profiler_ops,
        effective_tflops=effective_tflops,
        roofline=roofline,
    )


def summarize_run(
    *,
    case_name: str,
    checkpoint_path: str | Path,
    prepared_case: PreparedTrackerCase,
    num_iters: int,
    support_grid_size: int,
    seq_len: int,
    run_results: list[MinimalTrackerRunResult],
) -> dict[str, Any]:
    support_query_count = max((item.support_query_count for item in run_results), default=0)
    symbolic = build_symbolic_complexity_report(
        prepared_case=prepared_case,
        num_iters=num_iters,
        seq_len=seq_len,
        support_query_count=support_query_count or int(max(0, support_grid_size**2)),
    )
    return {
        "case_name": case_name,
        "checkpoint_path": str(checkpoint_path),
        "num_iters": int(num_iters),
        "support_grid_size": int(support_grid_size),
        "seq_len": int(seq_len),
        "symbolic_complexity": asdict(symbolic),
        "runs": [
            {
                **asdict(result),
                "roofline": asdict(result.roofline) if result.roofline is not None else None,
            }
            for result in run_results
        ],
    }
