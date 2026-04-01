# Xperience-10M Sample Tooling

This repository now contains a small maintained toolchain for the
`ropedia-ai/xperience-10m-sample` dataset.

## Scope

The tools target one local sample export that contains:

- six synchronized MP4 streams
- one `annotation.hdf5`
- caption segments
- depth, SLAM, full-body mocap, hand mocap, IMU, calibration, and metadata

The expected local dataset root can be passed with `--dataset-dir` or via
`XPERIENCE_SAMPLE_DIR`. On the current machine the default path is
`/DATA/disk0/shared/datasets/xperience-10m-sample`.

## Entrypoints

### Analyze the sample

```bash
python scripts/data_analysis/analyze_xperience_sample.py \
  --dataset-dir /DATA/disk0/shared/datasets/xperience-10m-sample \
  --output-dir data_tmp/xperience_sample_analysis
```

Outputs:

- `summary.json`
- `annotation_schema.tsv`
- `frame_probe_<idx>.json`

### Load aligned multimodal frames

```python
from scripts.data_analysis.xperience_sample_utils import XperienceSampleDataset

with XperienceSampleDataset("/DATA/disk0/shared/datasets/xperience-10m-sample") as dataset:
    sample = dataset.get_frame(
        2910,
        video_streams=("stereo_left", "stereo_right", "fisheye_cam1"),
        load_video=True,
        load_depth=True,
        load_mocap=True,
        load_imu=True,
        imu_radius=12,
    )
    print(sample.summary())
```

Returned per-frame fields include:

- `video_frames`
- `depth`, `depth_confidence`
- `slam_translation`, `slam_quat_wxyz`
- `full_body_keypoints`, `body_quats`, `contacts`
- `left_hand_joints`, `right_hand_joints`
- centered IMU windows
- active caption segment for the frame

### Render quick visualizations

```bash
python scripts/visualization/visualize_xperience_sample.py \
  --dataset-dir /DATA/disk0/shared/datasets/xperience-10m-sample \
  --output-dir data_tmp/xperience_sample_visualizations
```

Outputs:

- `storyboard.jpg`
- `dashboard_frame_*.jpg`
- `sample_animation.gif`

Each dashboard combines:

- one main RGB stream
- two synchronized secondary cameras
- depth and depth-confidence previews
- a top-down SLAM trajectory panel
- current caption segment and simple motion stats

## Notes

- Caption bounds are mixed: most segments use device timestamps, while the last
  segment in the sample uses a `frame_...` marker. The loader handles both.
- The caption config frame count and the actual stored frame count differ by
  one in the public sample. Downstream code should trust the stored arrays over
  the caption config.
- Generated artifacts should stay under `data_tmp/`; they are not source files.
