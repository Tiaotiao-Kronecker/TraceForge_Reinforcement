# TraceForge Extrinsics Fix Validation

Date: 2026-03-12

## Scope

This note records the code cleanup for the external-extrinsics direction bug and
the post-fix smoke validation on `press_one_button_demo_v1`.

## Code Changes

Updated files:

- `utils/extrinsics_utils.py`
- `utils/video_depth_pose_utils.py`
- `scripts/batch_inference/infer.py`
- `utils/test_extrinsics_utils.py`

Main logic changes:

1. Added a shared extrinsics utility module so external matrices are normalized
   in one place.
2. Unified wrapper output contract to `w2c`.
3. `VGGT4Wrapper` now converts VGGT `poses_pred` from `c2w` to `w2c` before
   returning.
4. `VGGT4Wrapper` external replacement path now also respects
   `--external_extr_mode`.
5. `ExternalGeomWrapper` still normalizes external inputs to `w2c`, but now
   uses the same shared helper.
6. Removed the unconditional `np.linalg.inv(extrs_npy)` in
   `scripts/batch_inference/infer.py`.

## Expected Post-Fix Contract

Inside the main inference pipeline:

- wrappers return `extrs_npy` as `w2c`
- downstream geometry consumes `w2c`
- saved output NPZ `extrinsics` remain `w2c`

## Smoke Validation

Dataset:

- `/data1/yaoxuran/press_one_button_demo_v1`
- `episode_00000`
- `camera = varied_camera_1`

### Test A: Correct mode

Command used:

```bash
env CUDA_VISIBLE_DEVICES=0 /home/wangchen/.conda/envs/traceforge/bin/python \
  /data1/wangchen/projects/TraceForge/scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path /data1/yaoxuran/press_one_button_demo_v1 \
  --out_dir /tmp/traceforge_extrinsics_fix_smoke \
  --camera_names varied_camera_1 \
  --episode_name episode_00000 \
  --checkpoint ./checkpoints/tapip3d_final.pth \
  --depth_pose_method external \
  --external_geom_name trajectory_valid.h5 \
  --external_extr_mode w2c \
  --device cuda:0 \
  --num_iters 1 \
  --fps 1 \
  --max_num_frames 16 \
  --horizon 8 \
  --frame_drop_rate 8 \
  --future_len 8 \
  --max_frames_per_video 8 \
  --grid_size 4
```

Result:

- output NPZ: `/tmp/traceforge_extrinsics_fix_smoke/episode_00000/varied_camera_1/varied_camera_1.npz`
- compared against:
  `trajectory_valid.h5 -> observation/camera/extrinsics/varied_camera_1`

Numeric check:

- direct diff: `0.0`
- inverse diff: `1.2027668952941895`

Interpretation:

- after the fix, `external_extr_mode=w2c` preserves the raw dataset extrinsics
  directly in the saved output
- this is the correct behavior for `press_one_button_demo_v1`

### Test B: Wrong mode, contrastive check

Same setup, but with:

- `--external_extr_mode c2w`

Numeric check:

- direct diff: `1.2027668952941895`
- inverse diff: `4.470348358154297e-08`

Interpretation:

- with the wrong mode, the saved output becomes the inverse of the raw H5
  extrinsics
- this confirms the old accidental double-inversion behavior is gone

## Final Conclusion

For `press_one_button_demo_v1`:

1. raw `trajectory_valid.h5` extrinsics should be interpreted as `w2c`
2. after this code fix, TraceForge correctly reads and uses them with
   `--external_extr_mode w2c`
3. `--external_extr_mode c2w` now behaves as a real alternative mode instead of
   accidentally canceling the old bug

## Comparison Against Last Night's 120-Episode Run

Checked logs under:

- `/data1/yaoxuran/press_one_button_demo_v1/_trajectory_batch_logs/worker_gpu*.log`

Observed behavior:

- the 120-episode run used `depth_pose_method=external`
- it also used `external_extr_mode='w2c'`

Representative log evidence:

- `worker_gpu7.log` line `15`:
  `[ExternalGeomWrapper] external_extr_mode='w2c'，按 world→camera 直接使用外参。`

### Direct output comparison on `episode_00000`

Old in-place output:

- `/data1/yaoxuran/press_one_button_demo_v1/episode_00000/trajectory`

New repaired rerun output:

- `/tmp/traceforge_extrinsics_fix_full_ep00000/episode_00000`

For all three cameras:

- new repaired output matches raw H5 extrinsics directly
- old in-place output matches the inverse of raw H5 extrinsics
- new repaired output is the inverse of the old in-place output

Numeric result:

- `varied_camera_1`
  - `new_vs_raw_direct = 0.0`
  - `old_vs_raw_inverse = 4.470348358154297e-08`
  - `new_vs_old_inverse = 0.0`
- `varied_camera_2`
  - `new_vs_raw_direct = 0.0`
  - `old_vs_raw_inverse = 2.9802322387695312e-08`
  - `new_vs_old_inverse = 0.0`
- `varied_camera_3`
  - `new_vs_raw_direct = 0.0`
  - `old_vs_raw_inverse = 5.960464477539063e-08`
  - `new_vs_old_inverse = 0.0`

Interpretation:

- the repaired `w2c` output is not numerically identical to last night's output
- it is the inverse-transform version of last night's output
- that is exactly what we expect after removing the old double-inversion bug

## Point-Cloud Check For Old vs New

Export root:

- `/data1/wangchen/projects/TraceForge/outputs_press_one_button_demo_v1_extrinsics_compare_2026-03-12`

Generated groups:

- `old_output_as_c2w`
- `new_output_as_w2c`
- `new_output_wrong_as_c2w`

Key result:

- `old_output_as_c2w` and `new_output_as_w2c` produce the same pairwise
  alignment metrics across the three cameras
- `new_output_wrong_as_c2w` is much worse

Correct interpretation pairwise metrics:

- `varied_camera_1__varied_camera_2`
  - chamfer `0.09968817234039307`
  - hit rate `0.07316666666666667`
- `varied_camera_1__varied_camera_3`
  - chamfer `0.2282252311706543`
  - hit rate `0.07133333333333333`
- `varied_camera_2__varied_camera_3`
  - chamfer `0.2858317643404007`
  - hit rate `0.036166666666666666`

Wrong interpretation pairwise metrics:

- `varied_camera_1__varied_camera_2`
  - chamfer `0.3227706775069237`
  - hit rate `0.0038333333333333336`
- `varied_camera_1__varied_camera_3`
  - chamfer `1.1218684315681458`
  - hit rate `0.0`
- `varied_camera_2__varied_camera_3`
  - chamfer `1.2453922033309937`
  - hit rate `0.0`

Interpretation:

- your previous Meshlab observation is consistent with the old output actually
  storing `c2w`
- after the fix, the new output stores `w2c`
- but if each result is interpreted using its true stored convention, the
  exported three-camera point clouds remain basically aligned
