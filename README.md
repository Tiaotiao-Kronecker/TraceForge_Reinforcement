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

### Single / generic inference (`vggt4 + cache`)

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

Use this path when RGB video or extracted RGB frames are available but depth /
camera geometry should be estimated by TraceForge itself.

Important: `infer.py` defaults to `--depth_pose_method vggt4`, but its default
`--scene_storage_mode source_ref` is currently only valid with
`--depth_pose_method external`. For VGGT-style runs, keep
`--scene_storage_mode cache` explicitly in the command.

### External geometry inference (`external + source_ref`)

```bash
python scripts/batch_inference/infer.py \
  --video_path <rgb_frames_dir> \
  --depth_path <depth_dir> \
  --external_geom_npz <trajectory_valid.h5_or_npz> \
  --camera_name <camera_name_if_h5> \
  --depth_pose_method external \
  --scene_storage_mode source_ref \
  --out_dir <output_dir> \
  --skip_existing \
  --frame_drop_rate 15 \
  --future_len 32 \
  --grid_size 80
```

Use this path when source depth, intrinsics, and extrinsics already exist.
`source_ref` keeps only references to source RGB / depth / geometry in
`scene_meta.json` instead of duplicating scene caches into each output episode.

### `infer.py` common parameters

#### Input and output

| Parameter | Default | Meaning | When to change |
| --- | --- | --- | --- |
| `--video_path` | required | Input RGB source. Can be a single video folder or a batch root when `--batch_process` is enabled. | Always set it. |
| `--out_dir` | `outputs` | Root directory for TraceForge outputs. Each processed video gets its own episode directory under this path. | Change when writing to a dataset-specific output root. |
| `--video_name` | `None` | Optional override for the output episode name. By default the script uses the input folder / video name. | Change only when you need stable custom naming. |
| `--batch_process` | `False` | Treat `--video_path` as a root directory and scan for multiple videos / frame folders. | Enable for dataset-style batch runs. |
| `--scan_depth` | `2` | How many directory levels below `--video_path` to scan when `--batch_process` is enabled. This is directory scanning depth, not trajectory depth. | Change when your dataset layout is shallower or deeper than the default two-level layout. |
| `--skip_existing` | `False` | Skip videos whose outputs are already complete. | Enable for resumable runs. |
| `--output_layout` | `v2` | Artifact layout to write. `v2` is the maintained format; `legacy` is only for compatibility. | Keep `v2` unless another tool explicitly requires legacy files. |

#### Geometry source and storage

| Parameter | Default | Meaning | When to change |
| --- | --- | --- | --- |
| `--depth_pose_method` | `vggt4` | How depth / intrinsics / extrinsics are obtained. `vggt4` estimates geometry; `external` loads known geometry. | Switch to `external` when the dataset already provides depth and camera geometry. |
| `--depth_path` | `None` | Path to known depth frames. Required by `--depth_pose_method external`. | Set it for external-geometry runs. |
| `--external_geom_npz` | `None` | Path to external intrinsics / extrinsics (`.npz` or `.h5`). With `external` it replaces the whole geometry stack; with `vggt4` it only replaces extrinsics. | Set it whenever external camera geometry is available. |
| `--camera_name` | `hand_camera` | Camera key used when `--external_geom_npz` points to an H5 file with multiple cameras. | Change for multi-camera H5 geometry, such as `varied_camera_1` or `varied_camera_3`. |
| `--external_extr_mode` | `w2c` | Declares whether external extrinsics are stored as world-to-camera (`w2c`) or camera-to-world (`c2w`). TraceForge internally uses `w2c`. | Change only if the source file stores `c2w`. |
| `--scene_storage_mode` | `source_ref` | Storage backend for `v2`. `source_ref` stores source paths plus frame mapping in `scene_meta.json`; `cache` writes local `scene.h5` and `scene_rgb.mp4`. | Use `source_ref` to avoid duplicating RGB / depth / geometry; use `cache` when the output must be self-contained. |
| `--save_visibility` | `False` | Save per-trajectory visibility arrays into each sample NPZ. | Enable only if downstream debugging or supervision needs visibility. |

#### Sampling and tracking

| Parameter | Default | Meaning | When to change |
| --- | --- | --- | --- |
| `--device` | `cuda` | Inference device. | Change to a specific GPU such as `cuda:0`, or to CPU only for debugging. |
| `--checkpoint` | `./checkpoints/tapip3d_final.pth` | TAPIP3D checkpoint path. | Change when testing another checkpoint or a non-default location. |
| `--num_iters` | `6` | Tracker refinement iterations. Larger values cost more runtime. | Usually keep default unless testing speed / quality tradeoffs. |
| `--fps` | `1` | Target sampling fps used by the preprocessing stage. | Change only if the source video fps handling needs a specific target. |
| `--max_num_frames` | `384` | Hard cap on frames loaded before later sampling / tracking logic. | Reduce for memory-constrained debugging, increase only when the model budget allows it. |
| `--horizon` | `16` | Sample horizon written into each saved segment. | Change only if downstream training expects a different segment horizon. |
| `--frame_drop_rate` | `1` | Query-frame stride. `1` means query every frame, `5` means every 5th frame, etc. It does not change the dense point cloud resolution. | Increase to reduce sample count and runtime. |
| `--future_len` | `128` | Tracking window length per query frame in offline mode. | Reduce for shorter clips / faster runs, increase when longer trajectories are useful. |
| `--max_frames_per_video` | `50` | Soft cap used when `--fps <= 0` to derive a stride from video length. | Change only for long-episode subsampling control. |
| `--grid_size` | `20` | Uniform query grid size per query frame. `20` means `20 x 20 = 400` query points. | Increase for denser tracks and stronger manipulator coverage; decrease for faster runs. |

#### Trajectory filtering

| Parameter | Default | Meaning | When to change |
| --- | --- | --- | --- |
| `--filter_level` | `standard` | Preset filtering strength. Higher levels are stricter about trajectory quality. | `standard` is the usual default; use `none` / `basic` for debugging, `strict` for cleaner but fewer tracks. |
| `--traj_filter_profile` | `external` | Camera/profile-specific filtering logic. This changes the rule shape, not just the strength. | Use wrist-oriented profiles for wrist cameras; keep `external` for non-wrist external cameras. |
| `--min_depth` | `0.01` | Minimum valid depth in meters. | Change only when the dataset uses a narrower near-field range. |
| `--max_depth` | `10.0` | Maximum valid depth in meters. | Lower it to reject very far noisy geometry; raise it for large scenes. |
| `--min_valid_frames` | `None` | Manual override for minimum valid frames per trajectory. `None` means follow the preset from `--filter_level`. | Change only for manual ablations or dataset-specific tuning. |
| `--visibility_threshold` | `None` | Manual override for the minimum visible-frame ratio. `None` means follow `--filter_level`. | Change only for manual filtering experiments. |
| `--boundary_margin` | `None` | Manual override for the image-edge safety margin in pixels. `None` means follow `--filter_level`. | Increase when edge-hugging projections are noisy. |
| `--depth_change_threshold` | `None` | Manual override for allowed depth volatility. `None` means follow `--filter_level`. | Lower it when depth jitter should be rejected more aggressively. |

Less frequently changed compatibility / debugging flags, such as
`--save_video` and `--use_all_trajectories`, are still available through
`python scripts/batch_inference/infer.py -h`.

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

This is the maintained dataset-facing batch entrypoint for
`press_one_button_demo_v1` style data. It assumes external depth and camera
geometry and writes TraceForge outputs per episode / camera.

| Parameter | Default | Meaning | When to change |
| --- | --- | --- | --- |
| `--base_path` | required | Dataset root that contains `episode_xxxxx/`, `rgb/`, `depth/`, and `trajectory_valid.h5`. | Always set it. |
| `--out_dir` | `None` | Optional global output root. When omitted, outputs are written in place under each episode. | Set it only if outputs should be separated from the source dataset. |
| `--trajectory_dirname` | `trajectory` | In-place output directory name under each episode when `--out_dir` is omitted. | Change only to keep compatibility with another local naming rule. |
| `--camera_names` | `varied_camera_1,varied_camera_2,varied_camera_3` | Comma-separated cameras to process. | Restrict it for partial reruns or camera-specific debugging. |
| `--episode_name` | `None` | Limit processing to one episode. | Set it for a single-case rerun. |
| `--gpu_id` | `None` | Comma-separated GPU list for built-in multi-GPU execution. | Set it for shared-machine or multi-GPU runs. |
| `--gpu_schedule_mode` | `dynamic` | Multi-GPU scheduling mode. `dynamic` uses a shared task queue; `static` keeps legacy sharding. | Keep `dynamic` unless reproducing an older static split. |
| `--depth_pose_method` | `external` | Geometry source for this dataset wrapper. | Keep default for press-one-button data. |
| `--external_geom_name` | `trajectory_valid.h5` | Per-episode geometry filename. | Change only if the geometry file name differs. |
| `--scene_storage_mode` | `source_ref` | `v2` storage backend. Default mode stores only source references plus frame mapping. | Switch to `cache` only when a self-contained local scene cache is required. |
| `--frame_drop_rate` | `15` | Query-frame stride. `15` reproduces the common `0/15/30/45` demo pattern. | Reduce for denser query coverage; increase for lighter outputs. |
| `--future_len` | `32` | Tracking window length for each query frame. | Increase for longer manipulation segments, decrease for faster debugging. |
| `--grid_size` | `80` | Query grid size. `80` means `6400` initial keypoints per query frame before filtering. | Lower it for faster verification runs. |
| `--filter_level` | `standard` | Preset filtering strength for saved `traj_valid_mask`. | Keep `standard` unless deliberately running a looser / stricter ablation. |
| `--traj_filter_profile` | `auto` | Camera-aware profile selection. `auto` maps wrist-like camera names to `wrist_manipulator_top95` and others to `external`. | Override only for explicit filter comparisons. |

Notes on `--traj_filter_profile`:

- `auto` maps names containing `wrist` / `hand`, or ending with `camera_3`, to
  `wrist_manipulator_top95`.
- `wrist_manipulator_top95` uses `wrist_manipulator` as the baseline and then
  drops the lowest-motion 5 percent of the remaining tracks per sample.
- Non-wrist cameras default to `external`.

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

`batch_infer.py` is an older wrapper for datasets already arranged like
`images0/` + `depth_images0/`. It is mainly a batch launcher around
`infer.py`, not the primary maintained entrypoint for all dataset formats.

| Parameter | Default | Meaning | When to change |
| --- | --- | --- | --- |
| `--base_path` | required | Root directory that contains trajectory folders such as `00000/images0` and `00000/depth_images0`. | Always set it. |
| `--out_dir` | `./output_traj_group0` | Root directory for wrapper-generated outputs. | Change for dataset-specific output placement. |
| `--frame_drop_rate` | `5` | Query-frame stride passed through to `infer.py`. | Increase for lighter runs, reduce for denser sampling. |
| `--gpu_id` | `None` | GPU list such as `0,1,2,3`. If omitted, the wrapper uses all available GPUs. | Set it to pin work to a subset of GPUs. |
| `--max_workers` | `None` | Maximum parallel workers. By default this follows the detected GPU count. | Lower it on memory-constrained machines or set `1` for serial debugging. |
| `--max_trajs` | `None` | Limit how many trajectories / episodes the wrapper processes. | Set it for quick tests. |
| `--test_mode` | `False` | Small-sample timing mode for estimating a full run. | Enable for throughput checks before full launch. |
| `--skip_existing` | `False` | Skip outputs that already exist. | Enable for resumable runs. |
| `--grid_size` | `None` | Optional override passed to `infer.py`. If omitted here, the wrapper does not override `infer.py`, so the effective grid size remains the `infer.py` default of `20`. | Set it when the wrapper run needs a denser or lighter query grid. |

### 3D visualization

#### Single-sample viewer

```bash
python scripts/visualization/visualize_single_image.py \
  --npz_path <episode_dir>/samples/<sample>.npz \
  --port 8080
```

`visualize_single_image.py` expects a sample NPZ path such as
`samples/<episode_name>_<query_frame>.npz`. By default it loads RGB, depth,
intrinsics, and extrinsics from the episode scene artifacts. `--image_path` and
`--depth_path` are optional overrides, not required inputs.

#### 3D animation viewer

Standard dynamic point-cloud view:

```bash
python scripts/visualization/visualize_3d_keypoint_animation.py \
  --episode_dir <episode_dir> \
  --query_frame 15 \
  --dense_pointcloud \
  --port 8080
```

Lighter-weight animation view:

```bash
python scripts/visualization/visualize_3d_keypoint_animation.py \
  --episode_dir <episode_dir> \
  --query_frame 15 \
  --dense_pointcloud \
  --keypoint_stride 2 \
  --dense_downsample 6 \
  --port 8080
```

If matplotlib cache permissions are restrictive on the current machine, run the
same commands as:

```bash
env MPLCONFIGDIR=/tmp/matplotlib python scripts/visualization/visualize_3d_keypoint_animation.py ...
```

Both `visualize_single_image.py` and
`visualize_3d_keypoint_animation.py` can read `v2` outputs produced with either
`scene_storage_mode=source_ref` or `scene_storage_mode=cache`.

#### `visualize_3d_keypoint_animation.py` parameters

| Parameter | Default | Meaning | When to change |
| --- | --- | --- | --- |
| `--episode_dir` | required | Episode root directory, not a sample NPZ file. The script reads `samples/` and scene artifacts from here. | Always set it. |
| `--query_frame` | `None` | Query frame to visualize. If omitted, the script uses the first available sample. If the requested frame is missing, it falls back to the first available sample. | Set it when comparing a specific query frame such as `15`. |
| `--keypoint_stride` | `10` | Display every Nth kept trajectory in the animation. Larger values render fewer keypoints and improve interactivity. | Increase it for dense samples or slow browsers. |
| `--dense_pointcloud` | `False` | Reconstruct and animate dense point clouds from scene artifacts. In `v2`, this follows `segment_frame_indices` and produces a dynamic background. | Enable it when depth / geometry consistency needs to be inspected. |
| `--dense_downsample` | `4` | Downsample factor for dense point clouds. Larger values produce fewer dense points. | Increase it to lighten rendering cost; decrease it for a denser background. |
| `--normalize_camera` | `False` | Transform trajectories and dense points into the query-frame camera coordinate system. | Enable it when camera motion should be factored out visually. |
| `--port` | `8080` | Local Viser port. | Change when the default port is occupied. |

Animation behavior notes:

- `--episode_dir` is the episode folder itself, unlike `visualize_single_image.py`
  which takes `--npz_path`.
- With `--dense_pointcloud`, `v2` outputs reconstruct dynamic dense point clouds
  directly from scene artifacts and source references.
- When the sample contains more than `500` kept trajectories and
  `--keypoint_stride` is greater than `1`, the script automatically resets the
  stride to `1` so all trajectories remain visible.

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
