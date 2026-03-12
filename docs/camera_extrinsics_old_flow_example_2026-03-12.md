# TraceForge Old External-Extrinsics Flow: Concrete Error Example

Date: 2026-03-12

## Final Conclusions

1. `press_one_button_demo_v1/trajectory_valid.h5` stores camera extrinsics in
   `w2c` format.
2. The old TraceForge `external` path had one incorrect global inversion after
   wrapper output and before downstream geometry/tracking.
3. Because of that bug, last night's 120-episode run was launched with
   `--external_extr_mode w2c`, but the saved output NPZ `extrinsics` ended up as
   the inverse of raw H5, namely `c2w`.
4. Old saved NPZ extrinsics can still produce aligned multiview point clouds if
   they are interpreted using their actual stored semantics, namely `c2w`.
5. Point-cloud alignment from old saved NPZ files does not prove the old
   TraceForge trajectory outputs are correct. The internal 3D tracking path had
   already consumed wrong-direction extrinsics before those NPZ files were
   written.
6. After the code fix, `press_one_button_demo_v1` is read correctly with
   `--external_extr_mode w2c`, and saved output NPZ `extrinsics` now match raw
   H5 directly in `w2c`.

## Purpose Of This Note

This note explains the old bug with one concrete example. It complements:

- `docs/camera_extrinsics_investigation_2026-03-12.md`
- `docs/camera_extrinsics_fix_validation_2026-03-12.md`

The main confusion was:

- why old exported point clouds could still look aligned in Meshlab
- but the old trajectory outputs were still not trustworthy

## One-Frame Toy Example

Assume raw dataset H5 stores the correct extrinsic as:

- raw H5 `w2c = T(-1, 0, 0)`

Its inverse is:

- true `c2w = T(+1, 0, 0)`

Interpretation:

- `w2c = T(-1, 0, 0)` means `x_cam = x_world - 1`
- `c2w = T(+1, 0, 0)` means `x_world = x_cam + 1`

Now take one 3D point that is seen by the camera at:

- camera-frame point `P_cam = (1, 0, 1)`

The correct world point should be:

- `P_world = c2w * P_cam = (2, 0, 1)`

## Old Buggy Flow

### Step 1. Raw H5 is loaded

For `press_one_button_demo_v1`, raw H5 stores:

- `extr_raw = w2c = T(-1, 0, 0)`

This part was correct.

### Step 2. `ExternalGeomWrapper` normalizes to `w2c`

User passed:

- `--external_extr_mode w2c`

So the wrapper kept the raw matrix unchanged:

- wrapper output `extr = w2c = T(-1, 0, 0)`

This part was also correct.

### Step 3. Old `infer.py` inverted once again

Old pipeline then applied a global inversion to wrapper output:

- `extr_after_infer = inv(w2c) = c2w = T(+1, 0, 0)`

This was the wrong step.

Reason:

- downstream TraceForge geometry utilities expect their input extrinsics to be
  `w2c`
- after this inversion, downstream instead received `c2w`

So the pipeline boundary became inconsistent exactly here.

## Why Downstream Geometry Then Became Wrong

Several downstream functions assume input extrinsics are `w2c`, then locally
invert them to recover `c2w` for unprojection.

For example, if downstream receives the already-wrong matrix:

- input to downstream = `c2w = T(+1, 0, 0)`

Then downstream does:

- `inv(c2w) = w2c = T(-1, 0, 0)`

and mistakenly treats that result as camera-to-world.

Applying that wrong transform to the camera point:

- wrong world estimate = `T(-1, 0, 0) * (1, 0, 1) = (0, 0, 1)`

But the correct world point should have been:

- correct world point = `(2, 0, 1)`

So the world coordinate is wrong by `2` units along `x`.

That is the essence of the old bug:

- raw H5 was correct
- wrapper normalization was correct
- the old global inversion between wrapper output and downstream geometry was
  incorrect

## Why Old Exported PLY Could Still Look Correct

The key is that old saved NPZ `extrinsics` were themselves written after the
wrong global inversion, so they were effectively stored as:

- saved NPZ `extrinsics = c2w`

If a later standalone point-cloud exporter reads those saved NPZ extrinsics and
interprets them as `c2w`, then it can still compute:

- `P_world = c2w * P_cam`

which is correct for point-cloud export.

So the old outputs can satisfy both statements at the same time:

1. their saved NPZ extrinsics can yield aligned PLY files if interpreted as
   `c2w`
2. their internal TraceForge 3D trajectory computation was still wrong because
   the tracking pipeline had already consumed the wrong-direction matrices

This is why the Meshlab observation was real, but it was not sufficient to
prove the old run was fully correct.

## What Happened In Last Night's 120-Episode Run

The run logs show it used:

- `depth_pose_method=external`
- `external_extr_mode='w2c'`

But because the old bug was still present at that time:

- raw H5 `w2c`
- wrapper kept `w2c`
- old `infer.py` inverted once more
- saved NPZ `extrinsics` became `c2w`

So the final saved extrinsics from that run are not in the same convention as
the raw H5 file, even though the launch mode was `w2c`.

## What We Verified Numerically

For `episode_00000` of `press_one_button_demo_v1`:

1. old in-place output matches the inverse of raw H5 extrinsics
2. fixed rerun output matches raw H5 extrinsics directly
3. fixed rerun output is the inverse of old in-place output

This means:

- old output NPZ `extrinsics` were effectively saved as `c2w`
- new fixed output NPZ `extrinsics` are saved as `w2c`

## Why Old Trajectories Are Still Not Trustworthy

Point-cloud export checks only one part of the story:

- whether saved depth plus saved extrinsics can be turned into mutually aligned
  world-space point clouds

But TraceForge trajectory generation depends on more than that:

- query lifting from image/depth to world coordinates
- repeated projection between world and camera coordinates during tracking

Those computations happened inside the old pipeline before results were saved.
If the pipeline used `c2w` where it expected `w2c`, then world-space trajectory
states were already corrupted.

That is exactly why old and fixed `coords` differ significantly even when old
PLY export can still look aligned under the correct interpretation of saved
NPZ semantics.

## Operational Interpretation

For `press_one_button_demo_v1`:

- raw `trajectory_valid.h5` should be treated as `w2c`
- fixed TraceForge should be run with `--external_extr_mode w2c`
- fixed saved output NPZ `extrinsics` should also be interpreted as `w2c`

For last night's old 120-episode outputs:

- saved NPZ `extrinsics` should be interpreted as `c2w` if you only want to
  export point clouds from those existing files
- but those old trajectory results should not be treated as final trustworthy
  curation outputs

## Bottom Line

The old bug was not in the raw dataset and not in the wrapper normalization. It
was the extra inversion inserted between wrapper output and downstream geometry.

That single wrong inversion explains all three observations together:

1. raw H5 for `press_one_button_demo_v1` is `w2c`
2. old saved NPZ extrinsics became `c2w`
3. old PLY export can still look aligned while old trajectory `coords` remain
   unreliable
