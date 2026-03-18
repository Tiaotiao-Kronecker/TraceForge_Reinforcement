# <img src="assets/trace_forge_logo.png" alt="TraceForge" height="25"> TraceForge

TraceForge is a dataset pipeline that turns videos plus camera geometry into
consistent 3D traces for robot learning.

Current maintained pipeline:

- depth and pose estimation or external geometry loading
- 3D point tracking
- sample serialization in `v2` layout by default
- trajectory filtering with camera-aware profiles
- visualization and verification on saved artifacts

The repository still contains a few compatibility paths, notably
`--output_layout legacy`. The old retarget helper is intentionally kept in code
for reference, but it is not active in the current inference/save path.

For model training on processed datasets, see
[TraceGen](https://github.com/jayLEE0301/TraceGen).

## Installation

```bash
conda create -n traceforge python=3.11
conda activate traceforge
bash setup_env.sh

mkdir -p checkpoints
wget -O checkpoints/tapip3d_final.pth \
  https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```

Run scripts from the repository root.

## Main Workflows

### Single / generic inference

```bash
python scripts/batch_inference/infer.py \
  --video_path <input_video_or_frames_dir> \
  --out_dir <output_dir> \
  --scene_storage_mode cache \
  --batch_process \
  --skip_existing \
  --frame_drop_rate 5 \
  --scan_depth 2 \
  --grid_size 20
```

### Press-one-button demo batch inference

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --camera_names varied_camera_1,varied_camera_2,varied_camera_3 \
  --depth_pose_method external \
  --external_geom_name trajectory_valid.h5 \
  --filter_level standard \
  --traj_filter_profile auto \
  --frame_drop_rate 15 \
  --future_len 32 \
  --grid_size 80
```

### Multi-GPU batch inference

```bash
python scripts/batch_inference/batch_infer.py \
  --base_path <dataset_root> \
  --out_dir <output_dir> \
  --gpu_id 0,1,2,3 \
  --skip_existing \
  --frame_drop_rate 5 \
  --grid_size 30
```

### 3D visualization

```bash
python scripts/visualization/visualize_single_image.py \
  --npz_path <episode_dir>/samples/<sample>.npz \
  --port 8080
```

`visualize_single_image.py` loads RGB/depth from the episode artifacts by
default. `--image_path` and `--depth_path` are optional overrides.

### Output verification

```bash
python checker/batch_process_result_checker_3d.py <output_dir> --max-videos 1 --max-samples 3
python checker/batch_process_result_checker.py <output_dir> --max-videos 1 --max-samples 3
```

## Output Layout

Default output is `v2` with `scene_storage_mode=source_ref`:

```text
<episode_dir>/
├── scene_meta.json
└── samples/
    ├── <sample0>.npz
    └── ...
```

`scene_meta.json` stores source RGB/depth/geometry references plus `source_frame_indices`.

When local scene caches are required, pass `--scene_storage_mode cache`; that variant also writes:

```text
<episode_dir>/
├── scene.h5
├── scene_rgb.mp4
```

Important `v2` sample fields:

- `traj_uvz`: query-camera `(u, v, depth)` trajectories
- `keypoints`: query-frame grid keypoints
- `query_frame_index`
- `segment_frame_indices`
- `traj_valid_mask`
- `traj_supervision_mask`
- optional `visibility`

`legacy` output is still supported behind `--output_layout legacy`, but it is a
compatibility mode rather than the recommended default.

See [docs/traceforge_output_structure.md](docs/traceforge_output_structure.md)
for maintained format details.

## Repository Guide

- [docs/README.md](docs/README.md): maintained technical documentation
- [docs/history/README.md](docs/history/README.md): archived investigations and
  validation notes
- [scripts/README.md](scripts/README.md): script index
- [CLAUDE.md](CLAUDE.md): contributor/agent repository guidance

## Notes

- Local experiment outputs under `data_tmp/` are intentionally ignored and not
  part of the repository interface.
- Historical investigation scripts live under
  `scripts/archived/investigations/`.
- If a document under `docs/history/` conflicts with current code, trust the
  maintained docs in `docs/`.
