# xperience stereo_left 成熟基线归因方案（2026-04-10）

## 目的

当前问题如果继续沿着：

- 裁边
- 改 ring
- 改 filter
- 再看 4D Viser

会很容易陷入“细节修补 + 反复观察视觉效果”的循环。

这份文档把后续排查收缩成一套更成熟、可复现、一次性定性的方案：

1. 先用成熟 pose baseline 替代当前 external extrinsics。
2. 再用现有 TraceForge 诊断脚本统一产出 JSON。
3. 最后给每个 case 一个明确标签，而不是继续做局部调参。

目标不是“让当前一版看起来最好”，而是给每个 case 一个主因结论：

- `00190 = depth boundary / bad seed`
- `00435 = pose 主导` 或 `depth 主导`
- `04234 = tracker / local background interaction`

## 当前执行进展

### 2026-04-12 更新：ORB-SLAM3 已接入并完成 `00435` 首轮跑通

本仓库内现已完成 ORB-SLAM3 的本地接入，具体位置如下：

- conda env: `.conda_envs/orbslam3`
- source tree: `third_party/orbslam3_src`
- binary: `third_party/orbslam3_src/Examples/RGB-D/rgbd_tum`
- helper script: `scripts/data_analysis/run_orbslam3_rgbd_case.py`

本轮为保证它能在远程机器上稳定离线运行，额外做了两处窄改动：

- `third_party/orbslam3_src/Examples/RGB-D/rgbd_tum.cc`
  - 关闭 Pangolin viewer，改成 headless
  - 去掉 realtime sleep，避免离线批处理空等
- `third_party/orbslam3_src/CMakeLists.txt`
  - `boost_serialization` 改为优先直连 conda 内的 `.so`

`00435` 当前已经实际跑通以下 ORB 分支产物：

- `geom/geom_stereo_left_orbslam3_rgbd_w2c.npz`
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd/stereo_left`
- `_analysis_mature_baseline/orbslam3_rgbd/prepare_summary.json`
- `_analysis_mature_baseline/orbslam3_rgbd/run_summary.json`
- `_analysis_mature_baseline/orbslam3_rgbd/conversion_summary.json`
- `_analysis_mature_baseline/orbslam3_scene_wobble.json`
- `_analysis_mature_baseline/tracker_geom_orbslam3_q0.json`
- `_analysis_mature_baseline/tracker_geom_orbslam3_q4.json`
- `_analysis_mature_baseline/triptych_orbslam3/summary_orbslam3.json`
- `_analysis_mature_baseline/triptych_orbslam3/composite_gifs/q00000_rgb_2d_query_static_3d.gif`
- `_analysis_mature_baseline/triptych_orbslam3/composite_gifs/q00004_rgb_2d_query_static_3d.gif`

其中 ORB 轨迹转换没有掉帧：

- `tracked_frame_count = 16`
- `missing_frame_indices = []`

`00435` 当前最关键的 baseline vs ORB 对比如下：

- `q0 global_final_disp_px: 2.565 -> 1.486`
- `q4 global_final_disp_px: 2.711 -> 1.839`
- `q0 residual_final_p95_px: 64.734 -> 52.792`
- `q4 residual_final_p95_px: 74.637 -> 52.519`
- `q0 track_final_p95_px: 63.950 -> 53.684`
- `q4 track_final_p95_px: 73.295 -> 54.123`

tracker-vs-geometry 侧也同步下降：

- `q0 tracker_final_drift_summary.median: 12.176 -> 5.457`
- `q4 tracker_final_drift_summary.median: 10.455 -> 5.680`
- `q0 tracker_local_interaction_count: 1149 -> 648`
- `q4 tracker_local_interaction_count: 1764 -> 1117`
- `q0 geometry_limited_count: 1762 -> 1253`
- `q4 geometry_limited_count: 1477 -> 1042`

这一步的中间结论现在比 2026-04-10 时更明确：

- `00435` 中 pose/extrinsics 明显参与了问题，而不是只有 depth 或 tracker 单边主导
- 但 ORB-SLAM3 替换后，尾部误差仍然还在 `50px+` 量级，所以它也不是“只要换 pose 就完全治好”
- 因此当前最合理的标签应更新为：
  - `00435 = pose 贡献明确，但仍残留 geometry tail + local interaction 的混合问题`

### `00435` 已完成的 baseline

已完成 case：

- `data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00435_officialprep`

已生成 baseline trajectory：

- `trajectory_dense_none_grid80_notrim_maturebase/stereo_left`

已生成分析目录：

- `_analysis_mature_baseline/`

其中当前已落盘的固定产物包括：

- `baseline_upstream.json`
- `baseline_scene_wobble.json`
- `tracker_geom_baseline_q0.json`
- `tracker_geom_baseline_q4.json`
- `triptych_baseline/summary_baseline.json`
- `triptych_baseline/composite_gifs/q00000_rgb_2d_query_static_3d.gif`
- `triptych_baseline/composite_gifs/q00004_rgb_2d_query_static_3d.gif`

### `00435` baseline 结论

当前 baseline 已经可以先给出一个中间结论：

- `00435` 不是“只有边缘少数坏点”的问题
- 上游 geometry-only 已经出现明显 heavy-tail wobble
- 但同时 tracker-vs-geometry 诊断也显示有不小的 local interaction

具体数值如下。

`baseline_upstream.json`：

- `q0 final_query_reproj_drift_median_px = 1.050`
- `q0 final_query_reproj_drift_p95_px = 77.848`
- `q4 final_query_reproj_drift_median_px = 0.662`
- `q4 final_query_reproj_drift_p95_px = 55.329`

这表示：

- 大部分点的典型误差不算特别夸张
- 但尾部点已经有几十像素级别的 scene wobble

`baseline_scene_wobble.json`：

- `q0 global_final_disp_px = 2.565`
- `q0 residual_final_p95_px = 64.734`
- `q0 track_final_p95_px = 63.950`
- `q4 global_final_disp_px = 2.711`
- `q4 residual_final_p95_px = 74.637`
- `q4 track_final_p95_px = 73.295`

这表示：

- full pipeline 中的背景残差漂移同样很重
- 问题已经不是单纯“query seed 不好看”

`tracker_geom_baseline_q0.json`：

- `tracker_final_drift_summary.median = 12.176`
- `tracker_final_drift_summary.p95 = 944.501`
- `static_geometry_final_drift_summary.median = 1.399`
- `excess_final_drift_summary.median = 5.789`
- `tracker_local_interaction_count = 1149`
- `geometry_limited_count = 1762`

`tracker_geom_baseline_q4.json`：

- `tracker_final_drift_summary.median = 10.455`
- `tracker_final_drift_summary.p95 = 1146.287`
- `static_geometry_final_drift_summary.median = 0.744`
- `excess_final_drift_summary.median = 6.314`
- `tracker_local_interaction_count = 1764`
- `geometry_limited_count = 1477`

这表示：

- `00435` 确实存在上游 geometry 问题
- 但也不能直接把全部责任都推给 geometry
- 至少在 `q4` 上，tracker/local interaction 已经和 geometry-limited 同量级

所以当前最合理的中间标签是：

- `00435 = geometry 明显参与，但尚未完成 pose vs depth 的最终拆分`

### 当前阻塞（已解除）

为了把 `00435` 继续拆成：

- `pose 主导`
- `depth 主导`
- `geometry 改善后剩余 tracker/local interaction`

下一步必须引入成熟 pose baseline。

到 2026-04-12 为止，这个阻塞已经解除：

- `ORB-SLAM3` 已在仓库内编通
- `ORBvoc.txt` 已解压可用
- `rgbd_tum` 已能 headless 运行
- `00435` ORB 分支已真实跑通并生成下游 TraceForge 结果

因此 `00435` 的后续主线被固定为：

1. 先接入 `ORB-SLAM3` 产出替代 extrinsics。
2. 用同一套 JSON 诊断脚本重跑。
3. 再决定是 `pose 主导`、`depth 主导`，还是 `geometry + tracker/local interaction` 混合问题。

## 选择的成熟基线

### 主基线：ORB-SLAM3

原因：

- 成熟度高
- 社区使用广
- 明确支持 `RGB-D / stereo`
- 最适合先回答 `00435` 这种 scene-level wobble 到底是不是 pose 主导

这里把 ORB-SLAM3 作为“替代 extrinsics”的主基线。

### 辅助交叉验证：COLMAP

原因：

- 非常成熟的离线 SfM 工具
- 适合做静态背景 pose 的离线交叉验证
- 不作为主线，只在 `00435` 上做可选 cross-check

这里把 COLMAP 作为“ORB-SLAM3 结果如果很反常时”的备用交叉验证。

### 不纳入主线的方案

本轮先不把这些纳入主线：

- 新研究型 video depth / point-map 工具
- 新 tracker 替换
- 再做新的 filter / trim ablation

原因很简单：

- 现在先要把 `geometry` 和 `tracker` 的责任边界划清
- 不是再引入一条新的研究路线

## 统一判定规则

### `00190`

如果满足：

- upstream geometry-only wobble 本来就不大
- 换 ORB-SLAM3 pose 后，scene-level wobble 没有本质改善
- 但错误轨迹仍主要集中在边缘 / 遮挡边界 / depth 很差的位置

则定性为：

- `depth boundary / bad seed`

### `00435`

如果满足：

- 换 ORB-SLAM3 pose 后，geometry-only 和 full pipeline 的 `global drift` 都显著下降
- `geometry_unstable` 明显减少

则定性为：

- `pose 主导`

如果满足：

- 换 ORB-SLAM3 pose 后改善很弱
- geometry-only 仍明显 wobble

则定性为：

- `depth 主导`

如果满足：

- geometry-only 已明显稳定
- 但 full pipeline 仍有大量背景漂移

则再回头定性为：

- `tracker / local interaction`

### `04234`

如果满足：

- baseline upstream 已较稳
- 换 ORB-SLAM3 pose 后 upstream 仍稳
- 但 tracker-vs-geometry 诊断中 `tracker_local_interaction_count` 持续远大于 `geometry_limited_count`

则定性为：

- `tracker / local background interaction`

## 统一输出目录

对每个 case，都统一产出到：

- `<case_dir>/_analysis_mature_baseline/`

其中固定使用：

- `baseline_upstream.json`
- `baseline_scene_wobble.json`
- `orbslam3_scene_wobble.json`
- `tracker_geom_baseline_q0.json`
- `tracker_geom_baseline_q4.json`
- `tracker_geom_orbslam3_q0.json`
- `tracker_geom_orbslam3_q4.json`
- `triptych_baseline/`
- `triptych_orbslam3/`

如果做 COLMAP cross-check，则额外产出：

- `colmap_scene_wobble.json`
- `triptych_colmap/`

## 统一环境变量

下面所有命令默认在仓库根目录运行：

```bash
cd /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience || exit 1
export MPLCONFIGDIR=/DATA/disk3/tmp/matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD
PY=/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python
CAM=stereo_left
QF=0,4
```

## 统一 baseline TraceForge rerun 命令

如果某个 case 还没有一套统一设置的 baseline output，先跑这个：

```bash
"$PY" scripts/batch_inference/infer.py \
  --video_path "$CASE_DIR/rgb/$CAM" \
  --depth_path "$CASE_DIR/depth/$CAM" \
  --external_geom_npz "$CASE_DIR/geom/geom_${CAM}_official_w2c.npz" \
  --camera_name "$CAM" \
  --video_name "$CAM" \
  --checkpoint ../TraceForge_Reinforcement/checkpoints/tapip3d_final.pth \
  --out_dir "$CASE_DIR/trajectory_dense_none_grid80_notrim_maturebase" \
  --depth_pose_method external \
  --external_extr_mode w2c \
  --scene_storage_mode source_ref \
  --fps 1 \
  --max_num_frames 512 \
  --future_len 32 \
  --frame_drop_rate 1 \
  --grid_size 80 \
  --support_grid_ratio 0.8 \
  --query_sampler_mode grid \
  --filter_level none \
  --traj_filter_profile external \
  --grid_border_trim_left 0 \
  --grid_border_trim_right 0 \
  --grid_border_trim_top 0 \
  --grid_border_trim_bottom 0 \
  --num_iters 5
```

baseline trajectory output 固定记作：

- `BASE_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_maturebase/$CAM"`

## 统一 baseline 诊断命令

### 1. upstream geometry-only wobble

```bash
ANA="$CASE_DIR/_analysis_mature_baseline"
mkdir -p "$ANA"

"$PY" scripts/data_analysis/export_external_wobble_upstream_report.py \
  --case_dir "$CASE_DIR" \
  --camera_name "$CAM" \
  --query_frames "$QF" \
  --grid_size 80 \
  --min_query_depth_m 0.2 \
  --min_border_dist_px 60 \
  --output_json "$ANA/baseline_upstream.json"
```

### 2. full pipeline scene wobble

```bash
"$PY" scripts/data_analysis/export_external_scene_wobble_report.py \
  --trajectory_output_dir "$BASE_TRAJ" \
  --query_frames "$QF" \
  --min_query_depth_m 0.2 \
  --min_border_dist_px 60 \
  --min_anchor_count 32 \
  --global_disp_threshold_px 3.0 \
  --output_json "$ANA/baseline_scene_wobble.json"
```

### 3. tracker-vs-geometry interaction

```bash
"$PY" scripts/data_analysis/export_tracker_geometry_interaction_report.py \
  --case_dir "$CASE_DIR" \
  --camera_name "$CAM" \
  --sample_npz "$BASE_TRAJ/samples/${CAM}_0.npz" \
  --output_json "$ANA/tracker_geom_baseline_q0.json"

"$PY" scripts/data_analysis/export_tracker_geometry_interaction_report.py \
  --case_dir "$CASE_DIR" \
  --camera_name "$CAM" \
  --sample_npz "$BASE_TRAJ/samples/${CAM}_4.npz" \
  --output_json "$ANA/tracker_geom_baseline_q4.json"
```

### 4. baseline triptych GIF

```bash
"$PY" scripts/visualization/compose_episode_triptych_gifs.py \
  --episode_dir "$CASE_DIR" \
  --trajectory_dirname "$(basename "$(dirname "$BASE_TRAJ")")" \
  --camera_name "$CAM" \
  --query_frames "$QF" \
  --output_dir "$ANA/triptych_baseline" \
  --render_mode finite \
  --2d_bg_mode static \
  --summary_name summary_baseline.json
```

### 5. baseline 4D Viser

```bash
"$PY" scripts/visualization/visualize_4d_reconstruction.py \
  --episode_dir "$BASE_TRAJ" \
  --render_mode finite \
  --buffer_mode buffered \
  --port 8091
```

## ORB-SLAM3 分支命令

### 前提

需要用户本机已有：

- `ORB_SLAM3_ROOT`
- 一份对应该相机的 ORB-SLAM3 settings yaml

下面约定：

```bash
ORB_SLAM3_ROOT=/path/to/ORB_SLAM3
ORB_RGBD_BIN="$ORB_SLAM3_ROOT/Examples/RGB-D/rgbd_tum"
ORB_VOC="$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt"
ORB_SETTINGS=/path/to/stereo_left_rgbd.yaml
ORB_OUT="$CASE_DIR/_analysis_mature_baseline/orbslam3_rgbd"
mkdir -p "$ORB_OUT"
```

### 1. 生成 ORB-SLAM3 association file

```bash
CASE_DIR="$CASE_DIR" CAM="$CAM" ORB_OUT="$ORB_OUT" python3 - <<'PY'
import os
from pathlib import Path

def sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (int(stem), stem)
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return (int(digits[-1]), stem)
    return (0, stem)

case_dir = Path(os.environ["CASE_DIR"])
cam = os.environ["CAM"]
orb_out = Path(os.environ["ORB_OUT"])
rgb_dir = case_dir / "rgb" / cam
depth_dir = case_dir / "depth" / cam
rgb_files = sorted(rgb_dir.glob("*"), key=sort_key)
depth_files = sorted(depth_dir.glob("*"), key=sort_key)
assert len(rgb_files) == len(depth_files), (len(rgb_files), len(depth_files))

assoc_path = orb_out / "associate.txt"
with assoc_path.open("w", encoding="utf-8") as f:
    for idx, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
        ts = f"{idx:.6f}"
        rgb_rel = rgb_path.relative_to(case_dir).as_posix()
        depth_rel = depth_path.relative_to(case_dir).as_posix()
        f.write(f"{ts} {rgb_rel} {ts} {depth_rel}\n")
print(assoc_path)
PY
```

### 2. 运行 ORB-SLAM3

```bash
cd "$ORB_OUT" || exit 1
"$ORB_RGBD_BIN" "$ORB_VOC" "$ORB_SETTINGS" "$CASE_DIR" "$ORB_OUT/associate.txt"
cd - >/dev/null || exit 1
```

预期产物：

- `"$ORB_OUT/CameraTrajectory.txt"`
- `"$ORB_OUT/KeyFrameTrajectory.txt"`

### 3. 把 ORB-SLAM3 轨迹转成 TraceForge 可读的 `geom_*.npz`

这里约定 ORB-SLAM3 输出是 TUM trajectory 格式：

- `timestamp tx ty tz qx qy qz qw`

转换命令：

```bash
CASE_DIR="$CASE_DIR" CAM="$CAM" ORB_OUT="$ORB_OUT" python3 - <<'PY'
import math
import os
from pathlib import Path
import numpy as np

def quat_xyzw_to_rot(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float32)

case_dir = Path(os.environ["CASE_DIR"])
cam = os.environ["CAM"]
orb_out = Path(os.environ["ORB_OUT"])
traj_path = orb_out / "CameraTrajectory.txt"
src_geom = case_dir / "geom" / f"geom_{cam}_official_w2c.npz"
dst_geom = case_dir / "geom" / f"geom_{cam}_orbslam3_rgbd_w2c.npz"

src = np.load(src_geom)
intrinsics = src["intrinsics"].astype(np.float32)
src.close()

rows = []
for line in traj_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) < 8:
        continue
    rows.append([float(x) for x in parts[:8]])

if len(rows) != len(intrinsics):
    raise RuntimeError(
        f"ORB-SLAM3 frame count mismatch: traj={len(rows)} vs intrinsics={len(intrinsics)}"
    )

extrinsics = np.zeros((len(rows), 4, 4), dtype=np.float32)
for i, row in enumerate(rows):
    _, tx, ty, tz, qx, qy, qz, qw = row
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = quat_xyzw_to_rot(qx, qy, qz, qw)
    c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    extrinsics[i] = np.linalg.inv(c2w).astype(np.float32)

np.savez(dst_geom, intrinsics=intrinsics, extrinsics=extrinsics)
print(dst_geom)
PY
```

### 4. 用 ORB-SLAM3 外参重跑 TraceForge

```bash
"$PY" scripts/batch_inference/infer.py \
  --video_path "$CASE_DIR/rgb/$CAM" \
  --depth_path "$CASE_DIR/depth/$CAM" \
  --external_geom_npz "$CASE_DIR/geom/geom_${CAM}_orbslam3_rgbd_w2c.npz" \
  --camera_name "$CAM" \
  --video_name "$CAM" \
  --checkpoint ../TraceForge_Reinforcement/checkpoints/tapip3d_final.pth \
  --out_dir "$CASE_DIR/trajectory_dense_none_grid80_notrim_orbslam3rgbd" \
  --depth_pose_method external \
  --external_extr_mode w2c \
  --scene_storage_mode source_ref \
  --fps 1 \
  --max_num_frames 512 \
  --future_len 32 \
  --frame_drop_rate 1 \
  --grid_size 80 \
  --support_grid_ratio 0.8 \
  --query_sampler_mode grid \
  --filter_level none \
  --traj_filter_profile external \
  --grid_border_trim_left 0 \
  --grid_border_trim_right 0 \
  --grid_border_trim_top 0 \
  --grid_border_trim_bottom 0 \
  --num_iters 5
```

ORB 分支 trajectory output 固定记作：

- `ORB_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_orbslam3rgbd/$CAM"`

### 5. ORB-SLAM3 分支诊断

```bash
"$PY" scripts/data_analysis/export_external_scene_wobble_report.py \
  --trajectory_output_dir "$ORB_TRAJ" \
  --query_frames "$QF" \
  --min_query_depth_m 0.2 \
  --min_border_dist_px 60 \
  --min_anchor_count 32 \
  --global_disp_threshold_px 3.0 \
  --output_json "$ANA/orbslam3_scene_wobble.json"

"$PY" scripts/data_analysis/export_tracker_geometry_interaction_report.py \
  --case_dir "$CASE_DIR" \
  --camera_name "$CAM" \
  --sample_npz "$ORB_TRAJ/samples/${CAM}_0.npz" \
  --output_json "$ANA/tracker_geom_orbslam3_q0.json"

"$PY" scripts/data_analysis/export_tracker_geometry_interaction_report.py \
  --case_dir "$CASE_DIR" \
  --camera_name "$CAM" \
  --sample_npz "$ORB_TRAJ/samples/${CAM}_4.npz" \
  --output_json "$ANA/tracker_geom_orbslam3_q4.json"

"$PY" scripts/visualization/compose_episode_triptych_gifs.py \
  --episode_dir "$CASE_DIR" \
  --trajectory_dirname "$(basename "$(dirname "$ORB_TRAJ")")" \
  --camera_name "$CAM" \
  --query_frames "$QF" \
  --output_dir "$ANA/triptych_orbslam3" \
  --render_mode finite \
  --2d_bg_mode static \
  --summary_name summary_orbslam3.json
```

### 6. ORB-SLAM3 分支 4D Viser

```bash
"$PY" scripts/visualization/visualize_4d_reconstruction.py \
  --episode_dir "$ORB_TRAJ" \
  --render_mode finite \
  --buffer_mode buffered \
  --port 8092
```

### 7. baseline vs ORB-SLAM3 JSON 对比

```bash
ANA="$CASE_DIR/_analysis_mature_baseline" python3 - <<'PY'
import json
import os
from pathlib import Path

ana = Path(os.environ["ANA"])
base = json.loads((ana / "baseline_scene_wobble.json").read_text())["rows"]
orb = json.loads((ana / "orbslam3_scene_wobble.json").read_text())["rows"]
base = {int(r["query_frame"]): r for r in base}
orb = {int(r["query_frame"]): r for r in orb}

for q in [0, 4]:
    print(f"QUERY {q}")
    for name, rows in [("baseline", base), ("orbslam3", orb)]:
        r = rows[q]
        print(name, {
            "geometry_unstable": r["geometry_unstable"],
            "global_final_disp_px": r["global_final_disp_px"],
            "residual_final_p95_px": r["residual_final_p95_px"],
            "track_final_p95_px": r["track_final_p95_px"],
        })
    print()
PY
```

## 可选 COLMAP cross-check

只建议在 `00435` 上做。目的不是替代主线，而是回答：

- 如果 ORB-SLAM3 结果和现有结论明显冲突，COLMAP 是否支持 ORB-SLAM3 的判断

### 1. 跑 COLMAP sparse reconstruction

```bash
COLMAP_OUT="$CASE_DIR/_analysis_mature_baseline/colmap_sparse"
mkdir -p "$COLMAP_OUT/sparse"

colmap feature_extractor \
  --database_path "$COLMAP_OUT/database.db" \
  --image_path "$CASE_DIR/rgb/$CAM" \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1

colmap sequential_matcher \
  --database_path "$COLMAP_OUT/database.db" \
  --SiftMatching.use_gpu 1

colmap mapper \
  --database_path "$COLMAP_OUT/database.db" \
  --image_path "$CASE_DIR/rgb/$CAM" \
  --output_path "$COLMAP_OUT/sparse"
```

### 2. 把 COLMAP sparse 结果转成 TraceForge `geom_*.npz`

```bash
CASE_DIR="$CASE_DIR" CAM="$CAM" COLMAP_OUT="$COLMAP_OUT" python3 - <<'PY'
import os
from pathlib import Path
import numpy as np
from datasets.utils.colmap import get_colmap_camera_params

def sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (int(stem), stem)
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return (int(digits[-1]), stem)
    return (0, stem)

case_dir = Path(os.environ["CASE_DIR"])
cam = os.environ["CAM"]
colmap_out = Path(os.environ["COLMAP_OUT"])
model_dir = colmap_out / "sparse" / "0"
img_files = sorted((case_dir / "rgb" / cam).glob("*"), key=sort_key)
K4, extr = get_colmap_camera_params(str(model_dir), [str(p) for p in img_files])
intr = K4[:, :3, :3].astype(np.float32)
extr = extr.astype(np.float32)
out_path = case_dir / "geom" / f"geom_{cam}_colmap_w2c.npz"
np.savez(out_path, intrinsics=intr, extrinsics=extr)
print(out_path)
PY
```

之后可完全复用 ORB-SLAM3 分支的 TraceForge rerun / wobble report / triptych 命令，只把：

- `geom_${CAM}_orbslam3_rgbd_w2c.npz`

替换成：

- `geom_${CAM}_colmap_w2c.npz`

并把输出目录名替换成：

- `trajectory_dense_none_grid80_notrim_colmap`

## Case 1：`00190`

### 目的

验证它是不是典型的：

- `depth boundary / bad seed`

而不是 pose 主导。

### 直接执行块

```bash
CASE_DIR=/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep
ANA="$CASE_DIR/_analysis_mature_baseline"
BASE_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_maturebase/$CAM"
ORB_OUT="$ANA/orbslam3_rgbd"
ORB_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_orbslam3rgbd/$CAM"
mkdir -p "$ANA"
```

按顺序执行：

1. 跑统一 baseline TraceForge rerun
2. 跑 baseline upstream / scene wobble / tracker-geometry / triptych
3. 跑 ORB-SLAM3 分支
4. 对比 `baseline_scene_wobble.json` 与 `orbslam3_scene_wobble.json`

### 本 case 最终需要保留的产物

JSON：

- `baseline_upstream.json`
- `baseline_scene_wobble.json`
- `orbslam3_scene_wobble.json`
- `tracker_geom_baseline_q0.json`
- `tracker_geom_baseline_q4.json`

图：

- `triptych_baseline/`
- `triptych_orbslam3/`

### 结论判定

如果 `00190` 的 ORB-SLAM3 分支：

- `global_final_disp_px` 变化很小
- `geometry_unstable` 没出现本质变化
- 但 triptych/4D 里错误轨迹依然主要集中在边缘和遮挡边界

就把 `00190` 定性为：

- `depth boundary / bad seed`

## Case 2：`00435`

### 目的

这是本轮最关键的 case，用来决定后续主线到底是：

- 继续修 pose
- 还是继续修 depth

### 直接执行块

```bash
CASE_DIR=/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00435_officialprep
ANA="$CASE_DIR/_analysis_mature_baseline"
BASE_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_maturebase/$CAM"
ORB_OUT="$ANA/orbslam3_rgbd"
ORB_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_orbslam3rgbd/$CAM"
mkdir -p "$ANA"
```

按顺序执行：

1. 跑统一 baseline TraceForge rerun
2. 跑 baseline upstream / scene wobble / triptych
3. 跑 ORB-SLAM3 分支
4. 跑 `baseline vs orbslam3` JSON 对比
5. 如果 ORB-SLAM3 结果异常，再跑 COLMAP cross-check

### 本 case 最终需要保留的产物

JSON：

- `baseline_upstream.json`
- `baseline_scene_wobble.json`
- `orbslam3_scene_wobble.json`
- 可选：`colmap_scene_wobble.json`

图：

- `triptych_baseline/`
- `triptych_orbslam3/`
- 可选：`triptych_colmap/`

### 结论判定

如果 ORB-SLAM3 分支相对 baseline：

- `global_final_disp_px` 明显下降
- `geometry_unstable` 明显减少
- 4D 里背景共同漂移明显减轻

则把 `00435` 定性为：

- `pose 主导`

如果 ORB-SLAM3 分支改善很弱，且 upstream/full pipeline 仍 wobble：

- `depth 主导`

如果 geometry-only 已明显稳定，但 full pipeline 仍然很差：

- `tracker / local interaction`

## Case 3：`04234`

### 目的

确认它是否真的是：

- `tracker / local background interaction`

而不是 upstream geometry 主导。

### 直接执行块

```bash
CASE_DIR=/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_04234_officialprep
ANA="$CASE_DIR/_analysis_mature_baseline"
BASE_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_maturebase/$CAM"
ORB_OUT="$ANA/orbslam3_rgbd"
ORB_TRAJ="$CASE_DIR/trajectory_dense_none_grid80_notrim_orbslam3rgbd/$CAM"
mkdir -p "$ANA"
```

按顺序执行：

1. 跑统一 baseline TraceForge rerun
2. 跑 baseline upstream / scene wobble / tracker-geometry / triptych
3. 跑 ORB-SLAM3 分支
4. 再跑 ORB 分支下的 tracker-geometry 诊断

### 本 case 最终需要保留的产物

JSON：

- `baseline_upstream.json`
- `baseline_scene_wobble.json`
- `orbslam3_scene_wobble.json`
- `tracker_geom_baseline_q0.json`
- `tracker_geom_baseline_q4.json`
- `tracker_geom_orbslam3_q0.json`
- `tracker_geom_orbslam3_q4.json`

图：

- `triptych_baseline/`
- `triptych_orbslam3/`

### 结论判定

如果 `04234` 在 baseline 和 ORB-SLAM3 两条分支里都满足：

- upstream geometry-only 已较稳
- `tracker_local_interaction_count` 仍远大于 `geometry_limited_count`
- 4D/triptych 中坏轨迹仍集中在局部背景区域

则把 `04234` 定性为：

- `tracker / local background interaction`

## 建议执行顺序

严格按下面顺序执行，不要并行开太多支线：

1. `00435`
2. `04234`
3. `00190`

原因：

- `00435` 最能决定后续主线该走 pose 还是 depth
- `04234` 最能决定是否需要替换 tracker
- `00190` 最像局部 bad depth / boundary 问题，放最后确认即可

## 这轮完成标准

只有满足下面三条，才算这一轮真的完成：

1. 三个 case 都有统一命名的 JSON 产物。
2. 三个 case 都至少有 baseline triptych 图。
3. 三个 case 都各自得到一个唯一主因标签，不再继续做 filter/trim 小修补。

## 2026-04-12 新主线：简化联合优化 `joint-lite-v1`

### 目标

- 不替换 tracker，不上完整 PointWorld pipeline。
- 只回答一个更具体的问题：
- 在 `ORB-SLAM3` pose baseline 之上，再联合稳定 `dense depth` 和剩余的小幅 `pose wobble`，能否把 `00435` 里残留的 `50px+` 级尾部误差继续压下去。

### 为什么这里只做简化版

- 当前 motion-window 工件本地只有 `rgb/stereo_left` 和 `depth/stereo_left`。
- 本地没有同级 `stereo_right`，也没有本地 `trajectory_valid.h5`。
- 因此这里不能直接做 PointWorld 那种完整的双目 + robot-state + visibility 联合优化。
- 这里的“联合”只指：
- 在当前 TraceForge inference 入口之前，交替改进 `extrinsics_w2c` 和 `depth_frames`，再把改进后的 geometry 送入原 inference。

### 这条主线的边界

- tracker 网络结构不改。
- model 权重不改。
- query sampler 先不改。
- `filter_level` 仍保持 `none`。
- 统一维持：
- `--filter_level none`
- `--query_sampler_mode grid`
- `--grid_size 80`
- 主初始 pose 使用 `ORB-SLAM3` 外参。
- 官方 `official_w2c` 只保留为 baseline 对照。

### 当前仓库里已经可复用的能力

- `scripts/data_analysis/run_orbslam3_rgbd_case.py`
- 作用：生成替代 `extrinsics_w2c` baseline。
- `utils/external_wobble_diagnostics.py`
- 已有：
- `smooth_extrinsics_w2c_moving_average`
- `freeze_extrinsics_w2c_to_query_frame`
- `estimate_temporal_median_world_points`
- `stabilize_depth_frames_temporal_median_reproject`
- `scripts/data_analysis/export_external_wobble_control_experiments.py`
- 作用：先做 geometry-only 控制变量实验。
- `scripts/batch_inference/infer.py`
- 已有：
- `--query_depth_stabilization_mode temporal_median_world_v1`
- `--dense_depth_stabilization_mode temporal_median_reproject_v1`

### `joint-lite-v1` 的变量定义

- `P_t`：第 `t` 帧相机外参 `extrinsics_w2c`
- `D_t`：第 `t` 帧 dense depth
- 先不直接优化 2D track。
- 先不做 full bundle adjustment。
- 先不在动态区域求解任何“真值”。

### `joint-lite-v1` 的算法定义

1. `P^(0)` 使用 `ORB-SLAM3` 外参。
2. `D^(0)` 使用原始 external depth。
3. `D-step`：
4. 在当前 `P^(k)` 下，用 `stabilize_depth_frames_temporal_median_reproject` 对整段 depth 做时序重投影中值稳定，得到 `D^(k+1)`。
5. `P-step`：
6. 在当前 `D^(k+1)` 固定时，对 `P^(k)` 只做小范围时序平滑，第一版先用 `smooth_extrinsics_w2c_moving_average(radius=1)` 得到 `P^(k+1)`。
7. 用 `(P^(k+1), D^(k+1))` 重跑原 TraceForge inference。
8. 第一轮先只做 `1` 次 alternating，不做多轮迭代。
9. 如果第一轮有效，再考虑：
10. `radius=2` 的 pose smooth
11. static-mask-aware 的 depth replace
12. 小范围的 pose residual refinement

### 为什么这个版本足够回答问题

- 如果 `ORB + dense depth stabilization` 明显变好，说明残余问题里 `depth` 仍占主要部分。
- 如果 `ORB + pose smooth` 明显变好，说明 `ORB` 之后仍残留可观的 `extrinsics wobble`。
- 如果只有两者一起上才明显变好，说明 residual 确实是 `depth / pose coupling`。
- 如果三者都改动不大，而 `tracker_local_interaction_count` 仍居高不下，就更像 `tracker / local background interaction`。

### `00435` 第一轮实验矩阵

- `A`: `official baseline`
- `B`: `ORB pose only`
- `C`: `ORB + query seed stabilization`
- 说明：这个分支已经存在，只作参考，不作为主判断依据。
- `D`: `ORB + dense depth stabilization`
- `E`: `ORB + pose smooth r1`
- `F`: `ORB + dense depth stabilization + pose smooth r1`
- 其中 `F` 定义为 `joint-lite-v1`。

### `00435` 第一轮命名约定

- `trajectory_dense_none_grid80_notrim_maturebase`
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd`
- `trajectory_dense_none_grid80_notrim_qdepthtmw1_densedepthtmr1`
- 这个目录已经存在，但它不是 ORB 主线，只能作参考。
- 计划新增：
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_densedepthtmr1`
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_extrsm1`
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_jointlitev1`

### `00435` 第一轮计划新增产物

- 目录：
- `_analysis_joint_lite/`
- JSON：
- `geom_controls_official_vs_orb_vs_jointlite.json`
- `scene_wobble_orbslam3rgbd_densedepthtmr1.json`
- `scene_wobble_orbslam3rgbd_extrsm1.json`
- `scene_wobble_orbslam3rgbd_jointlitev1.json`
- `tracker_geom_orbslam3rgbd_densedepthtmr1_q0.json`
- `tracker_geom_orbslam3rgbd_densedepthtmr1_q4.json`
- `tracker_geom_orbslam3rgbd_extrsm1_q0.json`
- `tracker_geom_orbslam3rgbd_extrsm1_q4.json`
- `tracker_geom_orbslam3rgbd_jointlitev1_q0.json`
- `tracker_geom_orbslam3rgbd_jointlitev1_q4.json`
- 图：
- `triptych_orbslam3rgbd_densedepthtmr1/`
- `triptych_orbslam3rgbd_extrsm1/`
- `triptych_orbslam3rgbd_jointlitev1/`

### `00435` 的核心判据

- 相对 `ORB pose only`，如果 `D` 明显优于 `E`：
- 残余主因更偏 `depth tail`
- 相对 `ORB pose only`，如果 `E` 明显优于 `D`：
- 残余主因更偏 `extrinsics residual wobble`
- 如果 `F` 明显优于 `D` 且也优于 `E`：
- 说明 `depth / pose coupling` 明确存在，后续值得做更正式的联合 refinement
- 如果 `F` 也只带来很弱改善，同时：
- geometry-only 已较稳
- `tracker_local_interaction_count` 仍高
- 4D 中错误主要留在局部背景区域
- 则把主因继续归到 `tracker / local background interaction`

### 量化成功标准

- 相对 `trajectory_dense_none_grid80_notrim_orbslam3rgbd`
- `global_final_disp_px` 不升高，最好继续下降
- `residual_final_p95_px` 继续下降
- `track_final_p95_px` 继续下降
- `tracker_final_drift_summary.median` 继续下降
- `geometry_limited_count` 继续下降
- `tracker_local_interaction_count` 不应显著升高
- 4D 里最重要的主观判据是：
- `00435` 背景整体共同漂移进一步减轻
- 而不是只让轨迹数量看起来变少

### 实现顺序

1. 新增一个导出脚本，把改进后的 `depth_frames` 和 `extrinsics_w2c` 写成独立 variant 资产。
2. 先在 `00435` 上产出 `ORB + dense depth stabilization`。
3. 再产出 `ORB + extrinsics smooth r1`。
4. 再产出 `joint-lite-v1`。
5. 用完全同一套 `scene wobble / tracker-geometry / triptych / 4D viser` 评估。
6. 只有当 `00435` 有明确增益后，才把同一方案复制到 `04234`。

### 暂不做的事

- 暂不引入新的 tracker。
- 暂不引入 full BA。
- 暂不引入 PointWorld 全流程。
- 暂不继续做新的 trim/filter 小修补。

### 本段落的真正目的

- 不是为了“调出一个更顺眼的 4D 可视化”。
- 而是要回答：
- `00435` 在 ORB 基线之后，剩余误差到底更像 `depth residual`、`pose residual`，还是 `tracker/local interaction residual`。

### 2026-04-12 第一轮真实结果

本轮已实际完成：

- `ORB + pose smooth r1`：
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_extrsm1`
- `ORB + dense depth stabilization (temporal_median_reproject_v1)`：
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_densedepthtmr1`
- `approx joint-lite`：
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_jointlitev1approx`

其中：

- `approx joint-lite` 的定义是：
- `extrsm1 geom + dense_depth_stabilization_mode=temporal_median_reproject_v1`
- 它不是严格的 `joint-lite-v1`，因为 dense stabilization 使用的是已平滑后的 pose，而不是先 ORB 再 dense 后再 smooth。
- 但它足够回答：
- “naive pose smooth 再叠 dense depth” 是否值得继续。

#### 1. `ORB + pose smooth r1` 结论

相对 `ORB pose only`：

- `q0 global_final_disp_px: 1.486 -> 3.606`
- `q0 residual_final_p95_px: 52.792 -> 57.201`
- `q0 track_final_p95_px: 53.684 -> 57.182`
- `q0 geometry_unstable: false -> true`
- `q4 global_final_disp_px: 1.839 -> 1.741`
- 但 `q4 residual_final_p95_px: 52.519 -> 56.658`
- `q4 track_final_p95_px: 54.123 -> 56.047`

tracker-vs-geometry 侧也恶化：

- `q0 tracker median: 5.457 -> 10.941`
- `q0 tracker_local_interaction_count: 648 -> 1200`
- `q0 geometry_limited_count: 1253 -> 1614`
- `q4 tracker median: 5.680 -> 10.398`
- `q4 tracker_local_interaction_count: 1117 -> 1450`
- `q4 geometry_limited_count: 1042 -> 1666`

因此第一条明确结论是：

- `naive extrinsics moving-average smoothing` 不是当前主线。
- 它没有修复 residual pose wobble，反而在 `00435` 上破坏了有效相机运动。

#### 2. `ORB + dense depth stabilization (temporal_median_reproject_v1)` 结论

相对 `ORB pose only`：

- `q0 global_final_disp_px: 1.486 -> 1.200`
- `q0 residual_final_p95_px: 52.792 -> 52.792`
- `q0 track_final_p95_px: 53.684 -> 53.616`
- `q4 global_final_disp_px: 1.839 -> 1.395`
- `q4 residual_final_p95_px: 52.519 -> 51.901`
- `q4 track_final_p95_px: 54.123 -> 53.122`

同时导出的稳定化资产 summary 表明 dense stabilization 的实际作用量不小：

- `dense_depth_stabilization.radius = 2`
- `dense_depth_stabilization.min_support = 3`
- `replace_ratio_median = 0.9722`
- `replace_ratio_p95 = 0.9935`
- `replace_count_total = 3734375`
- `depth_delta_median_median_m = 0.0095`
- `depth_delta_p95_p95_m = 0.5129`

因此第二条明确结论是：

- `00435` 在 ORB 基线之后，继续稳定 dense depth 是有效方向。
- 改善幅度虽然不算巨大，但方向是稳定正确的。

#### 2b. `densedepthtmr1` 严格 tracker-geometry 补跑（2026-04-13）

为避免只看 `scene wobble`，本轮已直接复用导出的稳定化资产：

- `_analysis_joint_lite/assets/orbslam3rgbd_densedepthtmr1/depth/stereo_left`
- `_analysis_joint_lite/assets/orbslam3rgbd_densedepthtmr1/geom/geom_stereo_left_orbslam3rgbd_densedepthtmr1_w2c.npz`

并确认资产是完整的：

- depth 帧数 `16 / 16`
- geom 文件 `1`
- rerun sample `8` 个

补跑命令如下：

```bash
"$PY" scripts/data_analysis/export_tracker_geometry_interaction_report.py \
  --case_dir "$CASE_DIR" \
  --sample_npz "$CASE_DIR/trajectory_dense_none_grid80_notrim_orbslam3rgbd_densedepthtmr1/$CAM/samples/${CAM}_0.npz" \
  --camera_name "$CAM" \
  --depth_dir "$CASE_DIR/_analysis_joint_lite/assets/orbslam3rgbd_densedepthtmr1/depth/$CAM" \
  --geom_npz "$CASE_DIR/_analysis_joint_lite/assets/orbslam3rgbd_densedepthtmr1/geom/geom_${CAM}_orbslam3rgbd_densedepthtmr1_w2c.npz" \
  --output_json "$CASE_DIR/_analysis_joint_lite/tracker_geom_orbslam3rgbd_densedepthtmr1_q0.json"

"$PY" scripts/data_analysis/export_tracker_geometry_interaction_report.py \
  --case_dir "$CASE_DIR" \
  --sample_npz "$CASE_DIR/trajectory_dense_none_grid80_notrim_orbslam3rgbd_densedepthtmr1/$CAM/samples/${CAM}_4.npz" \
  --camera_name "$CAM" \
  --depth_dir "$CASE_DIR/_analysis_joint_lite/assets/orbslam3rgbd_densedepthtmr1/depth/$CAM" \
  --geom_npz "$CASE_DIR/_analysis_joint_lite/assets/orbslam3rgbd_densedepthtmr1/geom/geom_${CAM}_orbslam3rgbd_densedepthtmr1_w2c.npz" \
  --output_json "$CASE_DIR/_analysis_joint_lite/tracker_geom_orbslam3rgbd_densedepthtmr1_q4.json"
```

相对 `ORB pose only`，严格版 `tracker-geometry` 对比如下。

`q0`：

- `tracker_final_drift_summary.median: 5.457 -> 4.919`
- `tracker_final_drift_summary.p95: 839.735 -> 824.724`
- `excess_final_drift_summary.median: 2.066 -> 1.779`
- `excess_final_drift_summary.p95: 81.927 -> 77.478`
- `tracker_local_interaction_count: 648 -> 498`
- 但 `static_geometry_final_drift_summary.median: 1.399 -> 1.788`
- `static_geometry_final_drift_summary.p95: 98.900 -> 106.137`
- `geometry_limited_count: 1253 -> 1270`

`q4`：

- `tracker_final_drift_summary.median: 5.680 -> 4.922`
- `tracker_final_drift_summary.p95: 1087.358 -> 893.110`
- `excess_final_drift_summary.median: 2.606 -> 1.936`
- `excess_final_drift_summary.p95: 80.336 -> 79.036`
- `tracker_local_interaction_count: 1117 -> 625`
- 但 `static_geometry_final_drift_summary.median: 0.744 -> 1.032`
- `static_geometry_final_drift_summary.p95: 59.778 -> 70.556`
- `geometry_limited_count: 1042 -> 1282`

worst-tail 的 `top_excess_tracks` 也说明同一个方向：

- `q0` top excess 基本仍由 `geometry_limited=true` 主导
- `q4` 虽然夹杂少量 `tracker_local_interaction=true`，但 top excess 仍主要是 `geometry_limited=true`

因此这一步把第二条结论进一步收紧为：

- `dense depth stabilization` 的收益主要落在 downstream residual / tracker-local interaction 侧。
- 它没有把 `static geometry` 的 heavy tail 一并治好，甚至在严格几何诊断上略有恶化。
- 所以 `00435` 当前最准确的表述不是“pose 还得继续平滑”，而是：
- `ORB 之后剩余问题以 depth residual 为主，并伴随未解的 geometry heavy tail`
- 下一步如果还要继续碰 pose，只能做带静态背景约束的 refinement，不能回到简单时序平均。

#### 3. `approx joint-lite` 结论

相对 `ORB + dense depth stabilization`：

- `q0 global_final_disp_px: 1.200 -> 3.221`
- `q0 residual_final_p95_px: 52.792 -> 55.829`
- `q0 track_final_p95_px: 53.616 -> 54.544`
- `q0 geometry_unstable: false -> true`
- `q4 global_final_disp_px: 1.395 -> 2.067`
- `q4 residual_final_p95_px: 51.901 -> 57.610`
- `q4 track_final_p95_px: 53.122 -> 57.366`

因此第三条明确结论是：

- 在当前 `00435` 上，把 `dense depth stabilization` 和 `naive pose smoothing` 直接叠加，不会带来联合收益。
- `pose smooth` 这一步当前更像是在伤害真实相机运动，而不是修复 residual wobble。

#### 第一轮总判断

截至这一轮，`00435` 的残余问题更像：

- `depth residual > pose residual`

更具体地说：

- `ORB-SLAM3` 已经把“大块 pose 问题”压下去了一轮。
- 剩余误差里，继续改 `dense depth` 有收益。
- 继续做 `naive pose smooth` 没收益，甚至有明显副作用。

因此当前主线应更新为：

- 暂停 `extrsm1` 方向。
- `densedepthtmr1` 资产导出与 strict tracker-geometry 诊断已完成。
- 后续主线继续放在 `depth residual` 与 `static geometry heavy tail` 的定向 refinement。
- 如果后面还要做 pose refinement，不能再用这种无约束时间均值；必须改成更强约束的静态背景一致性 refinement，而不是简单平滑。

### 2026-04-13 静态背景 pose refinement v1

本轮新增了以下工具与脚本：

- `utils/static_geometry_refinement.py`
- `scripts/data_analysis/export_static_geometry_heavy_tail_audit.py`
- `scripts/data_analysis/export_static_background_pose_refinement.py`
- `scripts/data_analysis/export_static_bg_refine_smoke_report.py`

并把 `export_external_wobble_upstream_report.py` 补成支持：

- `--depth_dir`
- `--geom_npz`

这样 refined geom 能直接复用同一套 upstream 诊断口径。

#### 1. heavy-tail audit 结果

先对 `ORB pose only` 做了 geometry heavy-tail audit：

- `q0 final drift p95 = 81.684`
- `q4 final drift p95 = 60.483`
- worst tail 明显集中在画面下沿：
- `y = 384~447`
- 重点 cell 为：
- `x = 128~191`
- `x = 256~319`
- `x = 320~383`
- `x = 384~447`

这说明当前 heavy tail 不是随机散点，而是稳定地落在底部大块区域。

#### 2. static-background pose refinement v1 定义

这一版不做 naive 时间平均，而是：

- 只用静态背景 anchor
- 在当前 pose 上冻结投影对应关系
- 用 world-to-camera 的 3D-3D rigid fit 求每帧小残差 pose
- 再只把 residual pose 当作主项，时序平滑最多只做弱正则

第一轮实际尝试了三组配置：

- `bgrefinev1`：
- `query_frames = [0, 4]`
- `temporal_regularization_weight = 0.25`
- `bgrefinev1_noreg`：
- `query_frames = [0, 4]`
- `temporal_regularization_weight = 0.0`
- `bgrefinev1_q0only`：
- `query_frames = [0]`
- `temporal_regularization_weight = 0.0`

#### 3. smoke gate 结果

smoke 固定复用：

- ORB trajectory sample：
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd/stereo_left/samples/stereo_left_0.npz`
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd/stereo_left/samples/stereo_left_4.npz`

gate 规则固定为：

- `global disp` 不得回退超过 `0.25px`
- `static_geometry_final_drift_summary.p95` 不得变差
- `geometry_limited_count` 不得变差

`bgrefinev1` 失败：

- `q0 upstream global: 0.153 -> 0.328`
- `q0 static geometry p95: 106.592 -> 107.259`
- 虽然 `q0 geometry_limited_count: 1341 -> 1271`
- 但 `static p95` 仍变差，所以不进 full rerun

`bgrefinev1_noreg` 也失败：

- `q0 upstream global: 0.153 -> 0.346`
- `q0 static geometry p95: 106.592 -> 107.315`

`bgrefinev1_q0only` 通过 smoke：

`q0`：

- `upstream global: 0.153 -> 0.114`
- `upstream drift p95: 81.684 -> 77.926`
- `static geometry p95: 106.592 -> 105.872`
- `geometry_limited_count: 1341 -> 1201`

`q4`：

- `upstream global: 0.154 -> 0.200`
- 仍在 `+0.25px` gate 内
- `upstream drift p95: 60.483 -> 54.351`
- `static geometry p95: 74.774 -> 70.307`
- `geometry_limited_count: 1409 -> 1382`

因此当前 static-background pose refinement 主线临时固定为：

- `bgrefinev1_q0only`

也就是：

- `query_frames = [0]`
- `temporal_regularization_weight = 0.0`

这说明在 `00435` 上：

- 静态背景 pose refinement 不是完全无效
- 但多 query + 轻时序正则会把 `q0` 拉坏
- 更局部、更保守的 pose residual refinement 才有机会带来净收益

#### 4. full rerun 真实结果

后续确认问题不是机器没 GPU，而是 sandbox 会话看不到 driver。

脱离 sandbox 后可见：

- `GPU 0~7 = NVIDIA H200`

其中当时空闲卡为：

- `GPU 0`
- `GPU 1`

因此已实际用 `GPU 0` 跑通：

- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_bgrefinev1`

对应 full pipeline scene wobble 结果如下。

相对 `ORB pose only`：

`q0`：

- `global_final_disp_px: 1.486 -> 0.933`
- `residual_final_p95_px: 52.792 -> 46.008`
- `track_final_p95_px: 53.684 -> 46.940`

`q4`：

- `global_final_disp_px: 1.839 -> 0.802`
- `residual_final_p95_px: 52.519 -> 44.096`
- `track_final_p95_px: 54.123 -> 44.804`

这说明 `bgrefinev1_q0only` 不只是 geometry-only smoke 变好，而是已经把 full pipeline 的 scene-level wobble 明确压下去了一轮。

严格版 tracker-geometry 结果如下。

相对 `ORB pose only`：

`q0`：

- `tracker median: 5.457 -> 5.199`
- `tracker p95: 839.735 -> 736.040`
- `excess median: 2.066 -> 1.079`
- `excess p95: 81.927 -> 65.170`
- `tracker_local_interaction_count: 648 -> 397`
- `geometry_limited_count: 1253 -> 1291`

`q4`：

- `tracker median: 5.680 -> 4.553`
- `tracker p95: 1087.358 -> 722.427`
- `excess median: 2.606 -> 1.669`
- `excess p95: 80.336 -> 58.825`
- `tracker_local_interaction_count: 1117 -> 436`
- `geometry_limited_count: 1042 -> 1403`

这里有一个关键分裂：

- full pipeline 指标显著改善
- tracker/excess 指标也显著改善
- 但 `geometry_limited_count` 没有同步下降，尤其 `q4` 反而升高

因此到这一刻最准确的结论是：

- `static-background pose refinement v1` 是有效的，而且比 `ORB pose only` 和 `densedepthtmr1` 都更强地压低了 full pipeline wobble
- 但它并没有把 strict geometry 分类里的 `geometry_limited` 一并消掉
- 换句话说，当前收益更像：
- `pose refinement` 改善了 full pipeline residual / tracker-vs-geometry excess
- 但底层 heavy-tail geometry label 仍没有被完全消解

和当前最强的 `densedepthtmr1` 相比：

- `bgrefinev1_q0only` 的 full pipeline 数值更好
- `q0 residual_final_p95_px: 52.792 -> 46.008`
- `q4 residual_final_p95_px: 51.901 -> 44.096`
- `q0 tracker_local_interaction_count: 498 -> 397`
- `q4 tracker_local_interaction_count: 625 -> 436`

所以 `00435` 当前主线应再更新一次：

- `static-background pose refinement` 不能停
- 但只保留 `q0only + no temporal regularization` 这个窄配置
- 它已经是当前最强 downstream variant
- 下一步不再纠结 naive smooth，也不再退回 `extrsm1`
- 应该直接考虑：
- `bgrefinev1_q0only + densedepthtmr1`
- 或者更强的 heavy-tail-aware depth refinement

### 2026-04-13 `bgrefinev1_q0only + densedepthtmr1` 联合结果

本轮已直接把 `bgrefinev1_q0only` 生成的 refined geom 和 `densedepthtmr1` 结合起来，完整导出了 joint dense assets，并完成 full rerun 与严格诊断。

关键产物如下：

- rerun output：
- `trajectory_dense_none_grid80_notrim_orbslam3rgbd_bgrefinev1_densedepthtmr1`
- joint dense assets：
- `_analysis_static_bg_refine_q0only/assets_joint_dense/orbslam3rgbd_densedepthtmr1/`
- asset summary：
- `_analysis_static_bg_refine_q0only/joint_dense_assets_summary.json`
- upstream：
- `_analysis_static_bg_refine_q0only/upstream_orbslam3rgbd_bgrefinev1_densedepthtmr1.json`
- scene wobble：
- `_analysis_static_bg_refine_q0only/scene_wobble_orbslam3rgbd_bgrefinev1_densedepthtmr1.json`
- strict tracker-geometry：
- `_analysis_static_bg_refine_q0only/tracker_geom_orbslam3rgbd_bgrefinev1_densedepthtmr1_q0.json`
- `_analysis_static_bg_refine_q0only/tracker_geom_orbslam3rgbd_bgrefinev1_densedepthtmr1_q4.json`

其中 dense depth 资产替换幅度很大：

- `replace_ratio_median = 0.970`
- `replace_ratio_p95 = 0.988`
- `replace_count_total = 3715330`

这说明该联合 variant 不是“轻微修补”，而是几乎全帧范围地用时序重投影结果重写了 depth。

#### 1. upstream geometry-only 对比

把口径统一到同一版 upstream 诊断之后，`ORB pose only`、`bgrefinev1_q0only`、`bgrefinev1_q0only + densedepthtmr1` 的对比如下：

`q0`：

- `ORB upstream global: 0.153`
- `bgrefine upstream global: 0.114`
- `joint upstream global: 0.081`
- `ORB upstream drift p95: 81.684`
- `bgrefine upstream drift p95: 77.926`
- `joint upstream drift p95: 78.577`

`q4`：

- `ORB upstream global: 0.154`
- `bgrefine upstream global: 0.200`
- `joint upstream global: 0.187`
- `ORB upstream drift p95: 60.483`
- `bgrefine upstream drift p95: 54.351`
- `joint upstream drift p95: 53.989`

这里的信号很明确：

- `joint` 没有把 upstream heavy tail 从根上再压掉一轮
- `q0` 甚至比 `bgrefine-only` 略回退：`77.926 -> 78.577`
- `q4` 只有很小的继续改善：`54.351 -> 53.989`

因此联合 variant 的主要收益来源，不应解释成“pose geometry 本身被进一步修好”，而更像是：

- `bgrefine` 先把 pose residual 拉回合理区间
- `densedepthtmr1` 再继续压 full pipeline 里的 depth residual / tracker interaction

#### 2. full pipeline scene wobble 对比

四组主要 downstream variant 结果如下：

`q0 global_final_disp_px`：

- `ORB only: 1.486`
- `dense only: 1.200`
- `bgrefine only: 0.933`
- `bgrefine + dense: 0.654`

`q0 residual_final_p95_px`：

- `ORB only: 52.792`
- `dense only: 52.792`
- `bgrefine only: 46.008`
- `bgrefine + dense: 41.844`

`q0 track_final_p95_px`：

- `ORB only: 53.684`
- `dense only: 53.616`
- `bgrefine only: 46.940`
- `bgrefine + dense: 42.372`

`q4 global_final_disp_px`：

- `ORB only: 1.839`
- `dense only: 1.395`
- `bgrefine only: 0.802`
- `bgrefine + dense: 0.843`

`q4 residual_final_p95_px`：

- `ORB only: 52.519`
- `dense only: 51.901`
- `bgrefine only: 44.096`
- `bgrefine + dense: 41.316`

`q4 track_final_p95_px`：

- `ORB only: 54.123`
- `dense only: 53.122`
- `bgrefine only: 44.804`
- `bgrefine + dense: 40.901`

这里可以直接下结论：

- `bgrefine + dense` 是当前最强的 full pipeline variant
- `q0` 上它相对 `bgrefine-only` 是全面继续改善
- `q4` 上 `global disp` 有轻微回退：`0.802 -> 0.843`
- 但 `residual p95` 和 `track p95` 继续明显下降，所以整体仍优于 `bgrefine-only`

#### 3. strict tracker-geometry 对比

相对 `bgrefine-only`，`bgrefine + dense` 的严格 tracker-geometry 结果如下。

`q0`：

- `tracker median: 5.199 -> 3.610`
- `tracker p95: 736.040 -> 623.219`
- `static geometry p95: 105.872 -> 105.857`
- `excess p95: 65.170 -> 65.163`
- `tracker_local_interaction_count: 397 -> 226`
- `geometry_limited_count: 1291 -> 1225`

`q4`：

- `tracker median: 4.553 -> 4.050`
- `tracker p95: 722.427 -> 697.739`
- `static geometry p95: 70.307 -> 63.132`
- `excess p95: 58.825 -> 63.445`
- `tracker_local_interaction_count: 436 -> 492`
- `geometry_limited_count: 1403 -> 1394`

这说明联合 variant 在 strict tracker-geometry 上不是“全指标单调更优”，而是：

- `q0` 基本全面继续改善
- `q4` 的 geometry side 继续改善
- 但 `q4 excess p95` 与 `tracker_local_interaction_count` 比 `bgrefine-only` 有小幅回退

不过如果参考 `ORB only`，它仍明显更强：

- `q4 tracker p95: 1087.358 -> 697.739`
- `q4 excess p95: 80.336 -> 63.445`
- `q4 tracker_local_interaction_count: 1117 -> 492`

所以更准确的说法不是“联合 variant 完全解决了 q4 交互问题”，而是：

- 它把大多数核心指标继续往正确方向推
- 但 `q4` 仍残留一部分 local interaction / excess tail

#### 4. 当前最终结论

到这一轮为止，`00435` 的结论应更新为：

- `approx joint-lite` 没有好处，这条线可以继续停
- `static-background pose refinement v1` 是有效主因之一
- 在它之上叠加 `densedepthtmr1` 后，得到当前最强 downstream variant
- 但联合 variant 的新增收益主要来自 `depth residual / tracker interaction`，不是 upstream pose geometry 再次被本质修复
- `00435` 当前最准确的标签应是：
- `ORB 之后仍有 static geometry heavy tail，但主剩余项更偏 depth residual + local interaction`

因此后续主线继续固定为：

- 保留 `bgrefinev1_q0only` 作为 pose refinement 基线
- 保留 `densedepthtmr1` 作为当前有效的 depth residual 压制手段
- 如需再往前推，不再做 naive smooth，也不回到 `extrsm1`
- 下一步应优先做：
- `heavy-tail-aware depth refinement`
- 或 `q4` 定向的 local interaction / residual tail 约束
