# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TraceForge converts cross-embodiment robot manipulation videos into consistent 3D traces via camera motion compensation and speed retargeting. The companion training repo is [TraceGen](https://github.com/jayLEE0301/TraceGen). Paper: [arXiv:2511.21690](https://arxiv.org/abs/2511.21690).

## Setup & Environment

```bash
conda create -n traceforge python=3.11 && conda activate traceforge
bash setup_env.sh                              # PyTorch 2.8.0 + CUDA 12.8 + all deps
cd third_party/pointops2 && python setup.py install  # CUDA extension, must match PyTorch CUDA version
```

Download checkpoint:
```bash
mkdir -p checkpoints
wget -O checkpoints/tapip3d_final.pth https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```

No `pyproject.toml` or `setup.py` at root — dependencies are installed via `setup_env.sh`.

## Key Commands

All commands run from the project root.

**Single video inference:**
```bash
python scripts/batch_inference/infer.py \
    --video_path <input_dir> --out_dir <output_dir> \
    --batch_process --use_all_trajectories --skip_existing \
    --frame_drop_rate 5 --scan_depth 2 --grid_size 20
```

**Multi-GPU batch inference:**
```bash
python scripts/batch_inference/batch_infer.py \
    --base_path <dataset_dir> --out_dir <output_dir> \
    --gpu_id 0,1,2,3 --use_all_trajectories --skip_existing \
    --frame_drop_rate 5 --grid_size 30
```

**3D trajectory visualization (viser):**
```bash
python scripts/visualization/visualize_single_image.py \
    --npz_path <output>/<video>/samples/<video>_0.npz \
    --image_path <output>/<video>/images/<video>_0.png \
    --depth_path <output>/<video>/depth/<video>_0.png --port 8080
```

**Validate output:**
```bash
python checker/batch_process_result_checker_3d.py <output_dir> --max-videos 1 --max-samples 3
python checker/batch_process_result_checker.py <output_dir> --max-videos 1 --max-samples 3
```

**VLM instruction generation:**
```bash
cd text_generation && python generate_description.py --episode_dir <dataset_dir> --skip_existing
```

## Architecture

### Pipeline Flow

```
RGB/Depth Video → Frame Extraction → Depth+Pose Estimation → 3D Point Query (grid) → TAPIP3D Tracking → 3D Traces (NPZ)
```

### Core Modules

- **`models/point_tracker_3d.py`** — `PointTracker3D` (nn.Module): the main TAPIP3D model. Orchestrates encoder → correlation features → point updaters. Handles 3D projection/unprojection via camera intrinsics/extrinsics.

- **`utils/inference_utils.py`** — `load_model()` and `inference()`: model loading and the `_inference_with_grid()` function that generates uniform grid queries and runs tracking.

- **`utils/video_depth_pose_utils.py`** — `BaseVideoDepthPoseWrapper` and the `video_depth_pose_dict` registry. Depth and camera pose estimation backends (default: `vggt4`). Handles depth scale alignment between predicted and known depth.

- **`datasets/datatypes.py`** — `SliceData` (torch tensors, for training) and `RawSliceData` (numpy arrays, for I/O). Core data containers holding RGB frames, depths, intrinsics, extrinsics, 3D trajectories, and visibility/validity masks. `SliceData` has `with_annot_mode('gt'|'est')` for accessing either ground-truth or estimated fields.

- **`datasets/data_ops.py`** — Composable data transforms via `register_transform` + `compose`. Used for training augmentations (crop, color jitter, etc.).

### Model Subsystems

- **`models/SpaTrackV2/`** — Video depth and pose estimation. Contains the VGGT4Track model (`vggt4track/`), SpaTrack modules, and a CoTracker-based tracker.
- **`models/moge/`** — MoGE (Mixture of Gaussian Experts) for 3D geometry.
- **`models/monoD/`** — Monocular depth models: DepthPro, Depth Anything V2, ZoeDepth.
- **`models/corr_features/`** — 4D correlation feature computation for point tracking.
- **`models/point_updaters/`** and **`models/encoders/`** — Encoder and point updater modules loaded by `PointTracker3D`.
- **`third_party/pointops2/`** — CUDA extension for point cloud operations (KNN, grouping, furthest sampling). **Known issue**: had hardcoded `cuda:0` tensor creation — fixed to use input tensor's device for multi-GPU support.
- **`third_party/cotracker/`** — 2D point tracking (dependency).

### Inference Scripts

- **`scripts/batch_inference/infer.py`** — Single-video processing. Contains `retarget_trajectories()` for synchronous arc-length retargeting that normalizes trajectories to a canonical speed.
- **`scripts/batch_inference/batch_infer.py`** — Multi-GPU parallel batch processing using subprocess-per-GPU.

### Key Arguments

| Arg | Purpose | Default |
|-----|---------|---------|
| `--scan_depth` | Directory nesting level (0=videos in root, 1=frames in subdirs, 2=frames in `images/` subdirs) | 2 |
| `--frame_drop_rate` | Query points every N frames | 1 |
| `--grid_size` | Keypoint grid resolution (grid_size^2 points/frame) | 20 |
| `--future_len` | Tracking window length per query | 128 |
| `--depth_pose_method` | Depth+pose backend (key in `video_depth_pose_dict`) | `vggt4` |
| `--support_grid_size` | Auto-computed as `grid_size * 0.8` | — |

### Depth Units

- Raw depth images (16-bit PNG) are typically in **millimeters**.
- Model-internal depth is in **meters**.
- Saved depth PNGs use **centimeters** (meters * 100, uint16, max 655.35m). Load by dividing by 100.
- For full precision, use `_raw.npz` files instead of PNG.

### Output Format

Each processed video produces:
- `images/` — Sampled RGB frames
- `depth/` — Depth maps (PNG + raw NPZ)
- `samples/` — Per-frame 3D trajectory NPZ files (the main output)
- `<video>.npz` — Full-video visualization data

## Language

README and troubleshooting sections are written in Chinese (中文). Code comments and docstrings are in English. The user communicates in Chinese.
