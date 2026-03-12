# Branch Note: Sim360 Extrinsics Fix Base

Date: 2026-03-12

## Branch Purpose

This branch is the working base for the next batch run on the 360 simulated
datasets.

It carries two core updates:

1. the camera-extrinsics direction bug fix in TraceForge
2. the dynamic multi-GPU camera-task scheduler for batch inference

## Extrinsics Fix Summary

### Problem

The old external-extrinsics path had a direction mismatch:

1. dataset `trajectory_valid.h5` extrinsics were already normalized in wrapper
   code
2. `scripts/batch_inference/infer.py` then inverted `extrs_npy` again
3. downstream geometry, which expects `w2c`, therefore consumed the wrong
   direction

This produced two confusing effects:

1. internal geometry and trajectory computation used the wrong extrinsics
   direction
2. saved output NPZ extrinsics from the old buggy path could still look usable
   for later point-cloud export if they were interpreted with their accidental
   stored semantics

### Fix

The current code fixes this by enforcing one internal contract:

- wrappers return `w2c`
- downstream geometry consumes `w2c`
- saved output NPZ extrinsics are also `w2c`

Implemented changes:

1. added `utils/extrinsics_utils.py` for shared normalization and inversion
2. `ExternalGeomWrapper` now normalizes all external inputs to `w2c` through
   one helper
3. `VGGT4Wrapper` converts VGGT `poses_pred` from `c2w` to `w2c`
4. removed the old unconditional inversion in
   `scripts/batch_inference/infer.py`

## Reason For The Fix

This fix is necessary because the previous behavior made point-cloud checks and
saved-NPZ checks misleading:

1. old outputs could still generate aligned PLY files under the accidental
   interpretation of stored extrinsics
2. that did not mean the internal trajectory computation was correct
3. the fixed code removes this ambiguity and makes raw dataset extrinsics,
   internal geometry, and saved outputs consistent

## Validated Result

Validated on:

1. `press_one_button_demo_v1`
2. DROID `trajectory_valid.h5`

Current operational conclusion:

- when the dataset H5 extrinsics are semantically `w2c`, run with
  `--external_extr_mode w2c`

Detailed evidence is recorded in:

- `docs/camera_extrinsics_fix_validation_2026-03-12.md`
- `docs/camera_extrinsics_old_flow_example_2026-03-12.md`
- `docs/droid_h5_w2c_single_case_validation_2026-03-12.md`

## Sim360 Usage Note

This branch is intended to be the code base for the upcoming run on the 360
simulated datasets.

For that use, the important expectation is:

1. external dataset extrinsics must be interpreted explicitly
2. the pipeline should stay on the fixed `w2c` internal contract
3. batch jobs should use the new dynamic GPU scheduling path when multi-GPU
   efficiency matters
