# Archived third-party code

These trees are kept for reference / benchmarking and are **not loaded at runtime**.
They are excluded from the maintained inference, batch, and training pipelines.

## spatrackv2/

SpatialTrackerV2 reference code (arXiv:2507.12462, ICCV 2025 — Xiao et al.).

- **Why here, not under `models/`**: TraceForge's runtime 3D point tracker is TAPIP3D
  (`models/point_tracker_3d.py::PointTracker3D`, arXiv:2504.14717). The SpaTrackerV2
  tree used to live under `models/SpaTrackV2/` but was never instantiated by
  `models.from_pretrained` / `from_config`; the only runtime edge in was a
  visualizer import inside `utils/viser_utils.py` whose caller was itself dead.
- **Internal imports still use `models.SpaTrackV2.*` paths**: intentional. The
  tree will not resolve as `models.SpaTrackV2` from the current package layout.
  Do not add sys.path shims to make it importable — if you need to compare
  against SpaTrackerV2, run it as a standalone subproject.
- **Checkpoint**: not needed for TraceForge. The runtime checkpoint is
  `checkpoints/tapip3d_final.pth` from `huggingface.co/zbww/tapip3d`.

If a future benchmark needs this code, either (a) vendor a fresh copy into a
dedicated benchmark harness, or (b) clone the upstream repo
(`github.com/henry123-boy/SpaTrackerV2`).
