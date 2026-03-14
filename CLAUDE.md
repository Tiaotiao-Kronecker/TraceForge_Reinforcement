# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TraceForge is a 3D trajectory tracking pipeline that converts cross-embodiment videos into consistent 3D traces via camera motion compensation and speed retargeting. The output is used to train TraceGen models for robotic manipulation.

**Key Concept**: The system tracks 400 query points (20×20 grid) across video frames, producing 3D world-coordinate trajectories with arc-length retargeting for consistent motion representation.

## Core Commands

### Setup
```bash
# Install dependencies (PyTorch 2.8.0 + CUDA 12.8)
bash setup_env.sh

# Download checkpoint
mkdir -p checkpoints
wget -O checkpoints/tapip3d_final.pth https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```

### Running Inference
```bash
# Single video
python infer.py --video_path <path> --out_dir outputs

# Batch processing with common settings
python infer.py \
    --video_path data/test_dataset \
    --out_dir outputs \
    --batch_process \
    --skip_existing \
    --frame_drop_rate 5 \
    --scan_depth 1
```

**Critical Parameters**:
- `--scan_depth`: Directory levels to scan (0=videos in root, 1=one subfolder, 2=two subfolders)
- `--frame_drop_rate`: Query points every N frames (1=every frame, 5=every 5th frame)
- `--future_len`: Tracking window length per query frame (default: 128)
- `--max_frames_per_video`: Target max frames to keep (default: 50)

### Visualization
```bash
# 3D trajectory viewer (opens viser web interface)
python visualize_single_image.py \
    --npz_path outputs/<video>/samples/<video>_0.npz \
    --image_path outputs/<video>/images/<video>_0.png \
    --depth_path outputs/<video>/depth/<video>_0.png \
    --port 8080

# Verify output files
python checker/batch_process_result_checker_3d.py outputs --max-videos 1 --max-samples 3
```

### Text Generation (VLM descriptions)
```bash
cd text_generation/
python generate_description.py --episode_dir <dataset_dir> --skip_existing
```
Requires `.env` file with `OPENAI_API_KEY` or `GOOGLE_API_KEY`.

## Architecture

### Pipeline Flow
```
Video Input → Depth/Pose Estimation → Query Point Generation →
3D Tracking (TAPIP3D) → Arc-Length Retargeting → NPZ Output
```

### Key Components

**1. Main Entry Point** (`infer.py`)
- `process_single_video()`: Core processing pipeline for one video
- `retarget_trajectories()`: Arc-length retargeting (lines 102-210)
- `save_structured_data()`: Saves output in TraceGen format (lines 212-393)

**2. Model Loading** (`models/__init__.py`)
- `from_pretrained()`: Loads TAPIP3D checkpoint
- `from_config()`: Creates model from Hydra config
- Model: `PointTracker3D` (Transformer-based 3D point tracker)

**3. Inference Utilities** (`utils/inference_utils.py`)
- `inference()`: Main inference function with support grid
- `get_grid_queries()`: Generates support points (16×16 grid)
- `_inference_with_grid()`: Adds support points to query points

**4. 3D Utilities** (`utils/threed_utils.py`)
- `project_tracks_3d_to_2d()`: Projects 3D trajectories to 2D
- `project_tracks_3d_to_3d()`: Transforms to fixed camera view
- Coordinate transformations between image/camera/world spaces

**5. Depth & Pose** (`utils/video_depth_pose_utils.py`)
- `video_depth_pose_dict`: Registry of depth/pose estimation methods
- Default: `vggt4` (MoGe-based depth + pose estimation)

### Data Flow Details

**Query Points**:
- 400 points per frame (20×20 uniform grid, hardcoded in line 284)
- Support points: 256 points (16×16 grid) added during inference
- Total processed: 656 points, but only 400 saved

**Coordinate Systems**:
1. Image coordinates (x, y) in pixels
2. Camera coordinates (X_cam, Y_cam, Z_cam)
3. World coordinates (X_world, Y_world, Z_world) - final output

**Arc-Length Retargeting** (`retarget_trajectories`):
- Converts variable-length trajectories to fixed length (default: 128 steps)
- Uses robust top-k mean for segment lengths (top 2%)
- Synchronous interpolation across all tracks
- Output: `(N, max_length, 3)` with `valid_steps` mask

## Output Structure

```
outputs/<video_name>/
├── images/<video>_<frame>.png          # Query frame RGB
├── depth/<video>_<frame>.png           # Depth visualization
├── depth/<video>_<frame>_raw.npz       # Raw depth values
├── samples/<video>_<frame>.npz         # Per-query-frame trajectories
└── <video_name>.npz                    # Full video data (first segment only)
```

**Sample NPZ Format** (for TraceGen training):
```python
{
    'image_path': str,              # Query frame path
    'frame_index': int,             # Query frame index
    'keypoints': (400, 2),          # Initial 2D positions
    'traj': (400, 128, 3),          # 3D trajectories (arc-length domain)
    'traj_2d': (400, T, 2),         # 2D projections (time domain)
    'valid_steps': (128,)           # Boolean mask for valid time steps
}
```

## Important Implementation Details

### Grid Size
- Query points: **20×20 = 400 points** (hardcoded, line 284)
- Support points: **16×16 = 256 points** (line 534)
- To modify grid size, update lines 284, 493, and 780

### Visibility Threshold
- Default: 0.9 (line 119 in `utils/inference_utils.py`)
- Points with `sigmoid(visib_logits) >= 0.9` marked as visible
- All 400 points saved regardless of visibility

### Memory Management
- Uses `torch.cuda.empty_cache()` between video segments
- Mixed precision inference with `torch.bfloat16`
- Processes query frames independently to limit memory

### Depth Processing
- Filters depth edges with `_filter_one_depth()` (0.08 threshold, 15 iterations)
- Uses IQR-based depth ROI for robust tracking
- Supports both estimated and known depth inputs

## Common Modifications

### Changing Query Point Density
Modify `grid_size` in three locations:
1. Line 284: `grid_size = 20` → `grid_size = 80`
2. Line 493: `grid_size=20` → `grid_size=80`
3. Line 780: Function default parameter

Note: 80×80 = 6400 points increases compute by 16×.

### Adjusting Tracking Window
```bash
--future_len 64  # Shorter tracking window (default: 128)
```

### Frame Sampling
```bash
--fps 0 --max_frames_per_video 100  # Auto-stride to keep ~100 frames
--fps 2                              # Fixed stride of 2
```

## Troubleshooting

**OOM Errors**: Reduce `--max_frames_per_video` or `--future_len`

**No trajectories**: Check `--scan_depth` matches directory structure

**Depth issues**: Verify depth units are consistent (meters vs millimeters)

**Slow inference**: Increase `--frame_drop_rate` to query fewer frames

## User-Created Documentation

⚠️ **Note**: The following are user-created analysis documents (in Chinese), not official documentation. They have been verified against official code (commit 2f0ce3c) with 100% accuracy.

### Core Technical Analysis
- `infer.py_详细讲解.md` - Detailed explanation of infer.py (functions, data flow, parameters)
- `docs/TraceForge数据格式与TraceGen训练分析.md` - Data format and TraceGen training analysis
- `docs/推理输出数据结构说明.md` - Inference output data structure specification
- `docs/文档验证报告.md` - Documentation verification report

### Architecture Deep Dives
- `support_grid_size作用机制详解.md` - Support grid mechanism (how 256 support points assist tracking)
- `查询点生成详解.md` - Query point generation (20×20 uniform grid details)
- `可见性掩码作用机制详解.md` - Visibility mask mechanism (0.9 threshold, all 400 points saved)

### Configuration & Troubleshooting
- `grid_size_80_修改指南.md` - Guide to modify grid_size from 20 to 80 (custom modification)
- `grid_size_vs_support_grid_size.md` - Difference between query and support grids
- `单位问题分析和修正.md` - Depth unit issues and fixes
- `错误来源和影响分析.md` - Error source and impact analysis
- `静止场景轨迹分析.md` - Static scene trajectory analysis

### Data Format & Usage
- `images0_npz说明.md` - NPZ file format explanation
- `可视化脚本适配说明.md` - Visualization script adaptation guide
- `batch_infer_traj_group0_使用说明.md` - Batch inference usage guide

## Related Projects

- **TraceGen**: https://github.com/jayLEE0301/TraceGen (model training)
- **Paper**: https://arxiv.org/abs/2511.21690
