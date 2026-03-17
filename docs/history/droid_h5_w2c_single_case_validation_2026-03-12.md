# DROID H5 `w2c` Single-Case Validation

> Historical note: this validation record is preserved as a point-in-time
> debugging artifact and is not maintained as current documentation.

Date: 2026-03-12

## Final Conclusion

For the DROID sample:

- `/data1/zoyo/projects/droid_preprocess_pipeline/droid_raw/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s`

the current fixed TraceForge code correctly reads:

- `trajectory_valid.h5 -> observation/camera/extrinsics/{camera}_left`

as semantic `4x4 w2c` extrinsics.

Under the fixed logic:

1. running inference with `--external_extr_mode w2c` produces output NPZ
   extrinsics that match raw H5 directly
2. exporting point clouds from those outputs with `w2c` interpretation gives
   better multiview alignment than the wrong `c2w` interpretation
3. unlike the old `press_one_button_demo_v1` bug, there is no hidden
   double-inversion cancellation here

## What Was Run

Episode:

- `/data1/zoyo/projects/droid_preprocess_pipeline/droid_raw/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s`

Cameras:

- `hand_camera`
- `varied_camera_1`
- `varied_camera_2`

Inference output root:

- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_droid_fixed_h5_w2c_case_2026-03-12/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s`

Inference mode:

- `depth_pose_method=external`
- `external_geom_npz=trajectory_valid.h5`
- `external_extr_mode=w2c`

## Numeric Extrinsics Check

Compared for frame `0`:

- output NPZ `extrinsics`
- raw H5 `observation/camera/extrinsics/{camera}_left`

Results:

- `hand_camera`
  - direct max abs diff: `0.0`
  - inverse max abs diff: `0.4088198244571686`
- `varied_camera_1`
  - direct max abs diff: `0.0`
  - inverse max abs diff: `0.4770887792110443`
- `varied_camera_2`
  - direct max abs diff: `0.0`
  - inverse max abs diff: `1.4643914699554443`

Interpretation:

- the saved output NPZ extrinsics are exactly the same matrices as the semantic
  DROID H5 extrinsics
- therefore the saved output NPZ convention here is `w2c`

## Point-Cloud Validation

Validation export root:

- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_droid_fixed_h5_w2c_case_2026-03-12/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s/firstframe_ply_validation_2026-03-12`

Two interpretations were exported from the same new inference outputs:

1. `output_as_w2c`
2. `output_wrong_as_c2w`

### Correct interpretation: `output_as_w2c`

- mean pairwise chamfer: `0.6735763673981031`
- mean pairwise hit rate: `0.015944444444444445`

Per-pair:

- `hand_camera__varied_camera_1`
  - chamfer `0.9860571324825287`
  - hit rate `0.003333333333333333`
- `hand_camera__varied_camera_2`
  - chamfer `0.5529574379324913`
  - hit rate `0.017166666666666667`
- `varied_camera_1__varied_camera_2`
  - chamfer `0.48171453177928925`
  - hit rate `0.027333333333333334`

### Wrong interpretation: `output_wrong_as_c2w`

- mean pairwise chamfer: `0.9017936487992605`
- mean pairwise hit rate: `5.555555555555555e-05`

Interpretation:

- the wrong `c2w` reading is clearly worse than the correct `w2c` reading
- so this case does not suffer from the old accidental “wrong semantics but
  still looks aligned” behavior

## Key Output Files

Summary JSON:

- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_droid_fixed_h5_w2c_case_2026-03-12/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s/firstframe_ply_validation_2026-03-12/summary.json`

Correct combined PLY:

- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_droid_fixed_h5_w2c_case_2026-03-12/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s/firstframe_ply_validation_2026-03-12/output_as_w2c/combined_frame00000.ply`

Wrong combined PLY:

- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_droid_fixed_h5_w2c_case_2026-03-12/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s/firstframe_ply_validation_2026-03-12/output_wrong_as_c2w/combined_frame00000.ply`

## Follow-Up Visualization Issue

This observation was made later on the full-parameter rerun:

- `/data1/wangchen/projects/TraceForge/data_tmp/history_outputs/outputs_droid_subset_fullparams_2026-03-12/AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s`

Using `scripts/visualization/visualize_3d_keypoint_animation.py` with:

- `--dense_pointcloud`
- `--query_frame 0`

the current qualitative result is:

1. the dense point-cloud 3D visualization looks too sparse and cannot assemble
   a complete scene
2. the 3D keypoint visualization, in contrast, can already reveal a much more
   complete scene structure

Current interpretation:

- this should be treated as a follow-up visualization / data-selection issue
- it does not overturn the current extrinsics-direction conclusion
- the dense point-cloud export and the interactive dense-pointcloud rendering
  path should be revisited later

Status:

- recorded for later investigation
- not resolved in this note

## Bottom Line

For DROID `trajectory_valid.h5`, if TraceForge reads:

- `observation/camera/extrinsics/{camera}_left`

then the correct mode under the fixed code is:

- `--external_extr_mode w2c`

and the resulting saved inference extrinsics should also be interpreted as
`w2c`.
