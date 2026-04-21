# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TraceForge is a unified dataset pipeline that converts cross-embodiment robotics videos into consistent 3D traces via camera motion compensation and speed retargeting. The maintained pipeline consumes external depth plus camera geometry, then runs 3D point tracking to generate trajectory data for robot learning.

**Key Paper**: arXiv:2511.21690 - TraceGen: World Modeling in 3D Trace Space

## Environment Setup

```bash
# Create and activate conda environment
conda create -n traceforge python=3.11
conda activate traceforge

# Install all dependencies (PyTorch 2.8.0 + CUDA 12.8)
bash setup_env.sh

# Download TAPIP3D checkpoint
mkdir -p checkpoints
wget -O checkpoints/tapip3d_final.pth https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```

**Important**: Always run scripts from the project root directory, not from subdirectories.

## Core Commands

### Single Video Inference
```bash
python scripts/batch_inference/infer.py \
    --video_path <input_video_directory> \
    --depth_path <input_depth_directory> \
    --external_geom_npz <trajectory_valid.h5_or_geom.npz> \
    --depth_pose_method external \
    --out_dir <output_directory> \
    --scene_storage_mode source_ref \
    --fps 1 \
    --max_num_frames 512 \
    --batch_process \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5 \
    --scan_depth 2 \
    --grid_size 20
```

### Sim / Button Batch Processing
```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
    --base_path <dataset_base_path> \
    --telemetry_out_dir <telemetry_root> \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --min_free_gpu_mem_gb 40 \
    --gpu_recovery_poll_sec 60 \
    --collect_profile_stats \
    --hardware_telemetry_interval_sec 30 \
    --depth_filter_workers 8 \
    --keyframes_per_sec_min 2 \
    --keyframes_per_sec_max 3 \
    --skip_existing
```

### 3D Visualization
```bash
python scripts/visualization/visualize_single_image.py \
    --npz_path <output_dir>/<video_name>/samples/<video_name>_0.npz \
    --port 8080
```

### Two-Hop Server Transfer
```bash
bash scripts/data_transfer/rsync_via_jump.sh \
    --src /data1/local_dataset/ \
    --dst /data/remote_dataset/ \
    --target-host <target_alias_or_host> \
    --ssh-config-file ~/.ssh/config \
    --tmux-session traceforge-transfer
```

This helper uses `rsync -aH --info=progress2 --partial --append-verify` over
SSH for resumable large-directory transfers. If your `target_host` alias already
contains `ProxyJump`, you can omit `--jump-host`; otherwise pass `--jump-host`
explicitly. Add `--print-only` when only checking the generated command,
`--dry-run` when doing remote file-list validation without copying data, and
`--bwlimit` if the link needs throttling. It defaults to `ssh -F /dev/null`;
pass `--ssh-config-file` if you need a custom SSH config.

### Output Verification
```bash
# 3D trajectory checker
python checker/batch_process_result_checker_3d.py <output_dir> --max-videos 1 --max-samples 3

# 2D trajectory checker
python checker/batch_process_result_checker.py <output_dir> --max-videos 1 --max-samples 3
```

### VLM Instruction Generation
```bash
cd text_generation/
python generate_description.py --episode_dir <dataset_directory> --skip_existing
```

## Architecture Overview

### Pipeline Flow
1. **Video Loading**: Load video frames from various dataset formats (raw videos, extracted frames, multi-level directories)
2. **Depth Loading**: Read external per-frame depth maps
3. **Camera Geometry Loading**: Read external intrinsics/extrinsics from H5/NPZ
4. **Query Frame Scheduling**: For button/sim episodes, build a shared per-episode query-frame schedule from `trajectory_valid.h5.attrs["fps"]`
5. **3D Point Tracking**: Track keypoints in 3D space using TAPIP3D model across video frames
6. **Output Generation**: Save 3D trajectories, depth maps, and metadata in NPZ format

### Key Components

**Models** (`models/`):
- `point_tracker_3d.py`: **Main runtime tracker** — `PointTracker3D`, the TAPIP3D implementation (Copyright TAPIP3D team, arXiv:2504.14717). Instantiated by `utils/inference_utils.py::load_model` via `models.from_pretrained` → `from_config` when loading `checkpoints/tapip3d_final.pth` (from `huggingface.co/zbww/tapip3d`).
- `corr_features/`: TAPIP3D Local Pair Attention / kNN feature-cloud correlation modules
- `point_updaters/`: 3D Trajectory Transformer iterative update blocks
- `encoders/`: RGB feature encoders
- `moge/`, `monoD/`: depth/geometry auxiliaries

**Archived (not runtime)**:
- `third_party/archived/spatrackv2/`: SpatialTrackerV2 reference code (arXiv:2507.12462). Not loaded by any runtime path; kept for comparison / benchmarking only. See `third_party/archived/README.md`.

**Inference Scripts** (`scripts/batch_inference/`):
- `infer.py`: Single/batch video inference with full pipeline
- `batch_infer_press_one_button_demo.py`: button/sim episode batch processing
- `batch_droid_external.py`: DROID external-only batch processing

**Utilities** (`utils/`):
- `inference_utils.py`: Model loading and inference orchestration
- `threed_utils.py`: 3D point transformations and projections
- `video_depth_pose_utils.py`: Depth/pose method registry
- `extrinsics_utils.py`: External extrinsics loading (H5/NPZ)
- `keyframe_schedule_utils.py`: Shared per-second query-frame sampling and raw/local index mapping

**Datasets** (`datasets/`):
- `data_ops.py`: Video/depth loading, preprocessing, filtering
- `datatypes.py`: Data structure definitions

### Critical Implementation Details

**Depth Unit Conversions**:
- Input depth images (16-bit PNG): typically in millimeters
- Model output depth: in meters
- Saved depth PNG: in centimeters (depth * 100.0, max 655.35m due to uint16)
- Always use `_raw.npz` files for full precision depth

**External Geometry Support**:
- Maintained mode is `--depth_pose_method external`
- `--external_geom_npz`: load external intrinsics + extrinsics from H5/NPZ
- `--external_extr_mode`: Specify if external matrices are `w2c` (world-to-camera, default) or `c2w` (camera-to-world)
- Supports both NPZ and H5 formats (use `--camera_name` for H5 multi-camera files)
- `--depth_path` is required in the maintained pipeline
- Default `scene_storage_mode=source_ref` stores references back to the source RGB/depth/geometry files

**Query Frame Sampling**:
- Generic `infer.py` uses `--frame_drop_rate` only when no shared schedule is provided
- `batch_infer_press_one_button_demo.py` samples shared keyframes per episode at `keyframes_per_sec_min~keyframes_per_sec_max` (default `2~3`)
- The maintained batch entry no longer exposes `--frame_drop_rate`, `--horizon`, or `--max_frames_per_video`
- To request a fixed per-second keyframe count in batch mode, set `keyframes_per_sec_min == keyframes_per_sec_max`
- Batch and benchmark entrypoints no longer hardcode maintained camera names; pass `--camera_names` / `--camera-names` explicitly, and only those cameras are processed
- Other maintained defaults still cover `depth_pose_method=external`, `external_geom_name=trajectory_valid.h5`, `fps=1`, `max_num_frames=512`, `future_len=32`, `num_iters=3`, `grid_size=80`, `filter_level=standard`, and `traj_filter_profile=external`
- Latest maintained external-only reference: `wipe_the_table_gs` median3 (`support_grid_ratio=0.8`, `varied_camera_1,varied_camera_2`, 2026-04-08) showed `num_iters=3` at about `1.52x` total-speedup vs `5`, with only mild systematic drift relative to `5` and no obvious extreme-case outliers; keep the maintained default throughput-first at `3`, while treating `num_iters=4` as the safer sanity-check fallback for new datasets or suspicious cases
- Next throughput-optimization backlog should target tracker forward rather than depth-filter: first try removing unnecessary CUDA synchronizations when `collect_profile_stats=False`, then profile low-risk same-semantics candidates such as hoisting repeated ROI/support-query prep out of the hot path, evaluating `torch.compile` / CUDA graph / TF32 on the maintained external-only benchmark, and trimming avoidable `clone` / `permute` / temporary-tensor churn in the tracker wrapper
- `batch_infer_press_one_button_demo.py` writes in-place to `<episode>/trajectory/<camera_name>/...` by default; passing `--out_dir` switches the root to `<out_dir>/<episode>/<camera_name>/...`
- `--telemetry_out_dir` decouples structured telemetry from artifact output: when set, `_batch_run_summary.json` and JSONL telemetry files are written there even if artifacts are still written in-place
- The true episode frame rate comes from `trajectory_valid.h5` root attr `fps`
- `--fps` is only the load stride; default `1`
- `--max_num_frames` is the post-stride frame cap; default `512`
- Source frame `0` is always forced into the shared query-frame schedule when it survives stride/cap/tail filtering
- This means the first second may contain one extra query frame beyond the nominal `keyframes_per_sec_min~max` sample count
- Query frames with `<= 8` remaining frames to the end of the loaded video segment (inclusive of the query frame itself) are discarded before schedule sampling
- `infer.py` still applies the same short-tail filter again when consuming either a shared schedule or the uniform-grid fallback, so stale manifests cannot reintroduce these frames
- For retained tail samples with `segment_len < future_len`, `v2` output pads the saved time axis to `future_len` with `inf` and marks the valid prefix via `valid_steps`
- Shared schedule manifests are written to `<episode_output>/_shared/query_frame_schedule_v3_<hash>.json`
- When the same episode is split across multiple batch submissions, pass the same `--shared_schedule_camera_names` set to each submission to force identical query-frame schedules
- `infer.py` consumes raw source-frame indices from the manifest, then maps them to the current local frame indices after stride/cap filtering

**Multi-GPU Processing**:
- Fixed `third_party/pointops2/functions/pointops.py` to use input tensor devices instead of hardcoded cuda:0
- `batch_infer_press_one_button_demo.py` supports one-command multi-GPU execution via `--gpu_id`
- The maintained multi-GPU path is dynamic-only
- Multi-GPU batch inference keeps a resident worker per GPU and schedules `episode/camera` tasks from a shared queue
- `--gpu_id` should list the actual currently usable physical GPU indices; the list may be sparse if some cards are unavailable (for example `1,3,5,6`)
- The shared query-frame schedule design keeps multi-camera keyframes aligned without breaking dynamic multi-GPU scheduling
- `--collect_profile_stats` writes task-level `profile_stats` / `save_profile_stats` into `_camera_task_profiles.jsonl` under `--telemetry_out_dir` when provided, otherwise under `--out_dir`
- `--hardware_telemetry_interval_sec > 0` writes periodic GPU/CPU/IO samples into `_hardware_telemetry.jsonl` under `--telemetry_out_dir` when provided, otherwise under `--out_dir`
- `--depth_filter_workers` controls `_DepthFilterRuntime` thread count for filtered-depth precomputation

**Keypoint Sampling**:
- `--grid_size N`: Samples N×N keypoints per frame (e.g., 20 = 400 points, 80 = 6400 points)
- `support_grid_size` auto-scales to `grid_size × 0.8`
- Higher grid_size increases computation time and memory usage

**Scan Depth Parameter**:
- `--scan_depth 0`: Videos directly in input folder
- `--scan_depth 1`: One subfolder per video with frames
- `--scan_depth 2`: Two-level layout (video_folder/images/)

## Dataset-Specific Notes

**DROID Dataset**:
- Use `batch_droid_external.py`
- H5 files contain camera extrinsics (w2c matrices) and depth (.npy files)
- Use `--external_geom_npz` to load H5 extrinsics
- Use `--camera_name` to specify camera (e.g., hand_camera, varied_camera_1)

**Button / Sim Episode Dataset**:
- Use `batch_infer_press_one_button_demo.py`
- Expected per-episode files include `trajectory_valid.h5`, `rgb/<camera>`, and `depth/<camera>`
- `trajectory_valid.h5` root attr `fps` drives the per-second query-frame schedule
- All cameras under one episode share the same raw query-frame indices
- `traj_filter_profile=auto` is retained only as a compatibility alias and currently resolves to `external`; wrist-oriented profiles remain available only when explicitly requested for historical investigations or compatibility reruns

**Sim360 Dataset**:
- Historical commit `ad064b0994c4` contains the Sim360 extrinsics fixes that previously lived on branch `curation/sim-360-extrinsics-fixed`
- See `docs/history/camera_extrinsics_investigation_2026-03-12.md` for details

## Common Issues

**CUDA Version Mismatch**:
- `pointops2` compilation may show CUDA version warnings (e.g., detected 13.1 but PyTorch uses 12.8)
- Modified `torch/utils/cpp_extension.py` to downgrade error to warning
- Compilation proceeds but use matching CUDA versions when possible

**Empty Output Directories**:
- Batch inference scripts now check output and warn if empty
- Use `--skip_existing` to avoid reprocessing
- Check logs for per-task file counts

**Depth Value Issues**:
- If depth values seem unreasonable (e.g., 100-600m for kitchen scenes), check unit conversions
- Verify depth loading code applies correct scaling (mm→m or cm→m)

## Output Structure

### Default layout: `v2 + source_ref`

```
<output_dir>/<video_name>/
├── scene_meta.json            # Layout metadata + source RGB/depth/geometry references
├── samples/                   # Per-frame 3D trajectories
│   ├── <video_name>_0.npz
│   ├── <video_name>_5.npz
│   └── ...
```

`source_ref` stores `source_frame_indices`, query-frame sampling metadata, plus
source RGB/depth/geometry paths in `scene_meta.json`, and reconstructs
frames/geometry lazily from those sources.

For button/sim episode outputs written in-place:

```
<episode_dir>/trajectory/
├── _shared/
│   └── query_frame_schedule_v3_<hash>.json
├── varied_camera_1/
├── varied_camera_2/
└── varied_camera_3/
```

If `--out_dir` is provided, the same per-episode structure is written under
`<out_dir>/<episode_name>/` instead of `<episode_dir>/trajectory/`.

Structured telemetry is written under `--telemetry_out_dir` when provided;
otherwise it falls back to `--out_dir`. If neither is set, only stdout logging
is available.

```
<telemetry_root>/
├── _batch_run_summary.json
├── _camera_task_metrics.jsonl
├── _camera_task_profiles.jsonl   # only when --collect_profile_stats
└── _hardware_telemetry.jsonl     # only when --hardware_telemetry_interval_sec > 0
```

When local scene caches are needed, pass `--scene_storage_mode cache`; that adds:

```
<output_dir>/<video_name>/
├── scene.h5                   # Shared per-frame depth/intrinsics/extrinsics cache
├── scene_rgb.mp4              # Shared RGB cache
```

`v2` sample NPZ contents:
- `traj_uvz`: query-frame camera coordinates in `(u, v, depth)` format, shape `(N_tracks, future_len, 3)` after tail padding
- `keypoints`: query-frame grid keypoints, shape `(N_tracks, 2)`
- `query_frame_index`: scalar query frame index
- `segment_frame_indices`: real video frame indices for the valid, non-padded prefix
- `valid_steps`: prefix-valid step mask over the saved time axis
- `traj_valid_mask`: per-trajectory validity mask
- `visibility`: optional per-trajectory visibility array when `--save_visibility` is enabled

### Legacy layout

Use `--output_layout legacy` to keep the old layout:

```
<output_dir>/<video_name>/
├── images/
├── depth/
├── samples/
└── <video_name>.npz
```

**⚠️ CRITICAL - NPZ Data Format**: Main NPZ and Sample NPZ use different coordinate/time semantics. See `docs/traceforge_output_structure.md` for the maintained format notes:
- Coordinate system differences (world coordinates vs image coordinates+depth)
- Time dimension handling (padding, segment lengths)
- Camera parameter coverage (full video vs segment)
- Visualization use cases and coordinate conversions

## Testing and Debugging

**Test Inference Output**:
```bash
python scripts/batch_inference/test_inference_output_shapes.py
```

**Verify Point Cloud**:
```bash
python scripts/batch_inference/verify_pointcloud.py
```

**Export PLY for Visualization**:
```bash
python scripts/visualization/export_pointcloud_ply.py
python scripts/visualization/export_ply_from_depth.py
```

**Check Failed Inference**:
```bash
python scripts/batch_inference/check_failed_inference.py
```

## Documentation

- `scripts/batch_inference/BATCH_INFERENCE_GUIDE.md`: Comprehensive batch processing guide
- `scripts/visualization/visualization_features.md`: Visualization capabilities
- `docs/traceforge_output_structure.md`: **Maintained output format and coordinate conventions**
- `docs/depth_scale_alignment_math.md`: Depth alignment mathematics
- `docs/history/README.md`: Historical investigations and archived notes
- `docs/history/camera_extrinsics_investigation_2026-03-12.md`: Historical camera extrinsics debugging

## Running Long Processes

Use `screen` to avoid interruption from network disconnects:

```bash
screen -S batch_inference
conda activate traceforge
python scripts/batch_inference/batch_infer_press_one_button_demo.py ...
# Detach: Ctrl+A then D
# Reattach: screen -r batch_inference
```
