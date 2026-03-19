# <img src="assets/trace_forge_logo.png" alt="TraceForge" height="25"> TraceForge

TraceForge is a dataset pipeline that turns videos plus camera geometry into
consistent 3D traces for robot learning.

Current maintained pipeline:

- external depth and camera geometry loading
- shared per-episode query-frame schedules for multi-camera button/sim datasets
- 3D point tracking
- sample serialization in `v2` layout by default
- trajectory filtering with camera-aware profiles
- visualization and verification on saved artifacts

The repository still contains a few compatibility paths, notably
`--output_layout legacy`. BridgeV2/VGGT-specific inference entrypoints have been
retired from the maintained workflow.

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
  --depth_path <input_depth_dir> \
  --external_geom_npz <trajectory_valid.h5_or_geom.npz> \
  --camera_name <camera_name_if_h5> \
  --out_dir <output_dir> \
  --scene_storage_mode source_ref \
  --fps 1 \
  --max_num_frames 512 \
  --batch_process \
  --skip_existing \
  --frame_drop_rate 5 \
  --scan_depth 2 \
  --grid_size 20
```

`infer.py` 的维护态是 external-only：必须提供 `--depth_path` 和
`--external_geom_npz`。`scene_storage_mode=source_ref` 是默认值；只有当输出必须
自带完整场景缓存时，才需要改成 `cache`。`--frame_drop_rate` 只在没有共享
schedule manifest 的 direct `infer.py` 路径里参与 query-frame 采样。

### Press-one-button demo batch inference

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --keyframes_per_sec_min 2 \
  --keyframes_per_sec_max 3 \
  --skip_existing
```

### Sim / button batch inference

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --gpu_id 0,1,2,3 \
  --min_free_gpu_mem_gb 40 \
  --gpu_recovery_poll_sec 60 \
  --keyframes_per_sec_min 2 \
  --keyframes_per_sec_max 3 \
  --skip_existing
```

For button/sim episode datasets, `trajectory_valid.h5` root attr `fps` is the
true episode frame rate used for shared per-second query-frame sampling.
`--fps` is only the RGB/depth/geometry load stride, and `--max_num_frames` is
the cap applied after that stride. The maintained batch entry is dynamic-only
in multi-GPU mode and no longer exposes `--frame_drop_rate`, `--horizon`, or
`--max_frames_per_video`; for a fixed per-second keyframe count, set
`--keyframes_per_sec_min` and `--keyframes_per_sec_max` to the same value.
Its maintained defaults already cover `camera_names=varied_camera_1,2,3`,
`depth_pose_method=external`, `external_geom_name=trajectory_valid.h5`,
`fps=1`, `max_num_frames=512`, `future_len=32`, `grid_size=80`,
`filter_level=standard`, and `traj_filter_profile=auto`.

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

`scene_meta.json` stores source RGB/depth/geometry references plus
`source_frame_indices` and query-frame sampling metadata.

For button/sim episode batches written in-place, the episode output root also
contains a shared keyframe manifest:

```text
<episode_dir>/trajectory/
├── _shared/
│   └── query_frame_schedule_v1_<hash>.json
├── varied_camera_1/
├── varied_camera_2/
└── varied_camera_3/
```

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
