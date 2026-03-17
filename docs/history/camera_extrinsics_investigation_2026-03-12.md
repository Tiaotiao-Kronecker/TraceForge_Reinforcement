# TraceForge Camera Extrinsics Investigation

> Historical note: this document records a 2026-03-12 investigation. It is
> kept for reference and may not match the current codebase. If referenced
> paths differ from the repository today, prefer `docs/` for maintained docs
> and `scripts/archived/investigations/2026-03/` for archived scripts.

Date: 2026-03-12

## Scope

This note records the current investigation results for camera extrinsics in the
TraceForge pipeline, with emphasis on:

- internal TraceForge extrinsics conventions
- `press_one_button_demo` and `press_one_button_demo_v1`
- current risks in the `external` depth/pose path
- previously verified DROID sample behavior

The goal is to separate confirmed facts from assumptions before modifying the
code.

## High-Level Conclusions

1. In the current TraceForge inference/output pipeline, the main saved
   `extrinsics` in `<video_name>.npz` should be interpreted as `w2c`
   (world-to-camera).
2. Several downstream geometry utilities also assume their input extrinsics are
   `w2c`.
3. There is a convention mismatch in the code:
   - the base wrapper docstring still says wrapper output extrinsics are `c2w`
   - `ExternalGeomWrapper` normalizes external inputs to `w2c`
   - `infer.py` later unconditionally inverts wrapper output once more
4. Because of that mismatch, the `external` path can silently flip extrinsics in
   the wrong direction, or accidentally become correct if the user passes the
   opposite `external_extr_mode`.
5. For the overlapping `episode_00000`, `press_one_button_demo` and
   `press_one_button_demo_v1` use the same intrinsics and extrinsics, but their
   RGB frames are different observations.
6. For that overlapping `episode_00000`, the raw dataset H5 extrinsics match the
   `outputs_press_one_button_demo_c2w` inference outputs directly and do not
   match their inverse. Combined with successful point-cloud stitching when
   reading those inference outputs as `w2c`, this supports the conclusion that
   the raw dataset stores `w2c`.
7. A previously checked DROID raw sample showed the opposite behavior: raw H5
   extrinsics matched `*_c2w` inference outputs, indicating that at least that
   DROID sample stores `c2w`.

## Code-Path Findings

### 1. Wrapper loading and normalization

Relevant files:

- `utils/video_depth_pose_utils.py`
- `scripts/batch_inference/infer.py`

Observed behavior:

- `_load_external_geom()` only loads matrices from NPZ/H5 and does not infer
  whether they are `w2c` or `c2w`.
- `ExternalGeomWrapper` explicitly normalizes external geometry to internal
  `w2c`:
  - `external_extr_mode='w2c'`: use directly
  - `external_extr_mode='c2w'`: invert once to obtain `w2c`
- `VGGT4Wrapper` can replace predicted extrinsics with external extrinsics, but
  that replacement path does not use `external_extr_mode`.

Important mismatch:

- `BaseVideoDepthPoseWrapper.__call__()` still documents wrapper output
  extrinsics as `camera-to-world`.
- That docstring is stale relative to newer `ExternalGeomWrapper`
  implementation.

### 2. Main inference pivot

Relevant file:

- `scripts/batch_inference/infer.py`

Critical step:

- After `model_depth_pose(...)` returns, `process_single_video()` does:

  `extrs_npy = np.linalg.inv(extrs_npy)`

This is the most important format pivot in the current pipeline.

### 3. Downstream geometry assumptions

Relevant files:

- `scripts/batch_inference/infer.py`
- `utils/common_utils.py`
- `utils/inference_utils.py`
- `models/point_tracker_3d.py`

Observed behavior:

- `prepare_query_points()` treats input `extrinsics[t]` as `w2c`, inverts it to
  `c2w`, then unprojects 2D pixels into world coordinates.
- `batch_unproject()` also assumes input extrinsics are `w2c`, because it
  computes `inv(extrinsic)` before mapping camera points to world points.
- `get_grid_queries()` likewise inverts input extrinsics before lifting points.
- `point_tracker_3d.py` applies `extrinsics` directly when projecting world
  coordinates into the camera frame, which is consistent with `w2c`.

### 4. Saved output convention

Relevant files:

- `scripts/batch_inference/infer.py`
- `docs/traceforge_output_structure.md`

Observed behavior:

- The main visualization NPZ stores:
  - `extrinsics = result["full_extrinsics"]`
- `full_extrinsics` is taken after the unconditional inversion in `infer.py`.
- The output structure doc also documents main NPZ `extrinsics` as `w2c`.

Conclusion:

- Main TraceForge output NPZ extrinsics should currently be interpreted as
  `w2c`.

## Current Risk in the `external` Path

### What the code currently does

`ExternalGeomWrapper` tries to normalize external inputs to `w2c`, but
`infer.py` later inverts wrapper output once again.

This means the code mixes two incompatible assumptions:

- old assumption: wrapper returns `c2w`, so `infer.py` should invert once
- newer assumption: external wrapper already returns `w2c`

### When the conflict happens

It mainly affects `--depth_pose_method external`.

If the raw dataset really stores `w2c`:

- user passes `--external_extr_mode w2c`
- wrapper keeps `w2c`
- `infer.py` inverts again
- downstream sees `c2w` where it expects `w2c`

If the raw dataset really stores `w2c`, but user passes
`--external_extr_mode c2w` by mistake:

- wrapper inverts raw `w2c` into `c2w`
- `infer.py` inverts once more
- output becomes `w2c` again by accident

So the current code can appear correct even when the mode flag is wrong.

### Practical consequence

Wrong-direction extrinsics will affect:

- 2D query lifting into world space
- dense depth unprojection into world points
- multiview point-cloud stitching
- main NPZ `extrinsics`

## Dataset Investigation: `press_one_button_demo` vs `press_one_button_demo_v1`

### Dataset relationship

Locations:

- `/data1/yaoxuran/press_one_button_demo`
- `/data1/yaoxuran/press_one_button_demo_v1`

Confirmed facts:

1. `press_one_button_demo` contains only `episode_00000`.
2. `press_one_button_demo_v1` contains `episode_00000` to `episode_00119`.
3. The only overlapping case is `episode_00000`.
4. For overlapping `episode_00000`:
   - RGB frames differ between the two datasets
   - intrinsics are identical
   - extrinsics are identical

Interpretation:

- The two datasets share the same camera geometry for `episode_00000`, but they
  are not the same RGB capture.

### Numeric comparison against inference outputs

Checked inference outputs:

- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_press_one_button_demo`
- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_press_one_button_demo_c2w`

Reference raw geometry:

- `/data1/yaoxuran/press_one_button_demo/episode_00000/trajectory_valid.h5`
- `/data1/yaoxuran/press_one_button_demo_v1/episode_00000/trajectory_valid.h5`

For both raw references, the numeric result is identical because their
intrinsics/extrinsics are identical.

#### A. `outputs_press_one_button_demo`

Max absolute difference between raw H5 extrinsics and output NPZ extrinsics:

- `varied_camera_1`: direct `1.2027669549`, inverse `3.3405933575e-08`
- `varied_camera_2`: direct `0.9087951109`, inverse `2.3149821504e-08`
- `varied_camera_3`: direct `0.5494972747`, inverse `2.9758656206e-08`

Conclusion:

- `outputs_press_one_button_demo` stores the inverse of the raw dataset H5
  extrinsics.

#### B. `outputs_press_one_button_demo_c2w`

Max absolute difference between raw H5 extrinsics and output NPZ extrinsics:

- `varied_camera_1`: direct `4.4703483582e-08`, inverse `1.2027669215`
- `varied_camera_2`: direct `2.9802322388e-08`, inverse `0.9087950886`
- `varied_camera_3`: direct `5.9604644775e-08`, inverse `0.5494972886`

Conclusion:

- `outputs_press_one_button_demo_c2w` matches the raw dataset H5 extrinsics
  directly.

### Interpretation for raw dataset format

We previously verified that point clouds built from
`outputs_press_one_button_demo_c2w` align across the three cameras when those
saved NPZ extrinsics are treated as `w2c` and inverted only at the final
unprojection step.

Since `outputs_press_one_button_demo_c2w` matches raw dataset H5 extrinsics
directly, this supports:

- raw `press_one_button_demo` / `press_one_button_demo_v1` H5 extrinsics are
  `w2c`

This means the `_c2w` in the output directory name should not be interpreted as
ground truth storage semantics. It is more likely a run-label that happened to
produce the correct final output because of the current double-inversion issue.

## Previously Verified DROID Result

Previously checked sample:

- `/data1/zoyo/projects/droid_preprocess_pipeline/droid_raw/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s/trajectory_valid.h5`

Confirmed result from earlier analysis:

- raw H5 extrinsics matched `*_c2w` inference outputs numerically
- raw H5 extrinsics did not match `*_w2c` inference outputs
- `*_w2c` outputs were the inverse of `*_c2w`

Interpretation:

- that DROID sample stores `c2w`

This is important because it means different datasets currently used with
TraceForge do not share the same raw extrinsics convention.

## Current Working Interpretation

At this point, the safest interpretation is:

1. TraceForge main output NPZ convention is `w2c`.
2. `press_one_button_demo` / `press_one_button_demo_v1` raw H5 convention is
   `w2c`.
3. At least one checked DROID raw H5 sample uses `c2w`.
4. Current `external`-mode code is not robust to dataset-specific convention
   differences because the inversion responsibilities are split incorrectly
   between wrapper and inference entrypoint.

## Recommended Next Steps

### Step 1: DROID raw-format audit

For `/data1/zoyo/projects/droid_preprocess_pipeline/droid_raw`:

- sample multiple episodes, not just one
- compare raw H5 extrinsics against both direct and inverse forms of inference
  output
- confirm whether DROID is uniformly `c2w` or mixed

### Step 2: TraceForge code cleanup

Recommended cleanup direction:

1. choose a single wrapper contract
   - preferred: all wrappers return `w2c`
2. remove the unconditional inversion in `infer.py`
3. update stale docstrings/comments
4. ensure `VGGT4Wrapper` external replacement path obeys the same
   `external_extr_mode` handling as `ExternalGeomWrapper`
5. add a minimal regression test:
   - raw external `w2c` input should remain `w2c` in saved output
   - raw external `c2w` input should be converted once and only once
   - multiview point clouds should align under the correct convention

## Summary

The current evidence does not support the claim that
`press_one_button_demo`-family raw extrinsics are actually `c2w`. The evidence
instead supports:

- raw `press_one_button_demo`-family H5 extrinsics are `w2c`
- current TraceForge `external` handling contains a double-inversion bug
- a wrongly chosen mode flag can accidentally cancel that bug
- DROID raw extrinsics likely differ from the press dataset and need separate
  dataset-level confirmation before code changes are finalized
