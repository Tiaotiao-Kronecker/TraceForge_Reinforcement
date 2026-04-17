# xperience qvis near-exempt 0.8m 对照与暂用结论（2026-04-16）

## 背景

对象仍是：

- `data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep`

原当前过滤版：

- `trajectory_dense_none_grid80_notrim_qvisv1_fixuvdv1_trajuvdv1/stereo_left`

在 4D 可视化里，近处手臂区域存在明显“被过滤掉”的现象。为区分是前验 gate 还是后验 gate 在误杀，这一轮补做了：

1. 原前验后验过滤版
2. `qvis off + fixuvdv1 + trajuvdv1`
3. `qvis near-exempt 0.8m + fixuvdv1 + trajuvdv1`

统一设置保持一致：

- `frame_drop_rate=1`
- `future_len=32`
- `grid_size=80`
- `support_grid_ratio=0.0`
- `grid_border_trim_left/right/top/bottom=0`
- `filter_level=none`
- `traj_filter_profile=external`
- query frames 仍为 `0..7`

## 核心结论

### 1. 近处手臂缺失，主因是 pre-track `qvis`

对 `trajectory_dense_none_grid80_notrim_qvisv1_fixuvdv1_trajuvdv1` 的 sample 直接统计可见：

- `q0` 中 query depth `<0.5m` 的点共 `840`
  - 被 `future visibility` 删掉 `538`
  - 被 `fixed-view depth gate` 删掉 `7`
  - 被后验 `traj_uvd` 删掉 `0`
- `q4` 中 query depth `<0.5m` 的点共 `811`
  - 被 `future visibility` 删掉 `443`
  - 被 `fixed-view depth gate` 删掉 `0`
  - 被后验 `traj_uvd` 删掉 `0`

因此这类“近处手臂消失”不是后验 `traj_uvd` 主导，而是前验 `all_future_v1` 在 tracking 前就把这些 query 挡掉了。

### 2. `qvis off` 和 `near-exempt 0.8m` 都能把近处手臂放回来

按 8 个 query frame 汇总的 final valid 统计：

- 原过滤版：`19941`
- `qvis off`: `29656`
- `qvis near-exempt 0.8m`: `27807`

其中近处点恢复量几乎一样：

- 原过滤版 `near<0.5m` valid：`3108`
- `qvis off`：`6819`
- `qvis near-exempt 0.8m`：`6819`

### 3. `qvis off` 的副作用明显更大

按 8 个 query frame 汇总：

- 原过滤版 `far>=0.8m` valid：`10701`
- `qvis off`：`12602`，比原版多 `1901`
- `qvis near-exempt 0.8m`：`10758`，比原版只多 `57`

解释：

- `qvis off` 不只是恢复近处手臂，也把大量中远处点一起放回来了
- `near-exempt 0.8m` 近处恢复效果和 `qvis off` 接近，但中远处分布基本仍接近原过滤版

## 当前决策

对当前 `xperience stereo_left 00190` 的 4D 可视化对照，先使用：

- `trajectory_dense_none_grid80_notrim_qvisv1_nearexempt08_fixuvdv1_trajuvdv1`

原因：

- 它保住了近处手臂
- 它的副作用明显小于直接关闭 `qvis`
- 它更适合作为当前 case 的工程折中

但这不是维护态默认结论，只是当前 xperience case-local 暂用方案。

## 泛化判断

`near-exempt 0.8m` 仍然是 heuristic，不应被误读成“无阈值方案”。

它的问题是：

- 它依赖绝对米制阈值
- 它不是“手臂豁免”，而是“所有 `<0.8m` 的 query 都豁免”
- 换相机安装位姿、FOV、工作距离或深度尺度后，`0.8m` 可能过松或过紧

因此当前判断是：

- 在这批 `xperience stereo_left` case 上，`near-exempt 0.8m` 可先用
- 若后续要推广为更稳的维护态方案，不应直接把 `0.8m` 视为最终默认值

更值得做的长期方向是：

- 不让 `qvis` 单独做 pre-track hard kill
- 把 `qvis` 降成 soft score / 联合判据的一部分
- 与 `fixuvd` / `traj_uvd` 一起决定最终删除，而不是由 `all_future_v1` 一票否决

## 新增开关

本轮为 `qvis` 新增了近距豁免开关：

- `--query_visibility_gate_near_depth_exempt_threshold_m`

实现位置：

- `utils/query_visibility_gate_utils.py`
- `scripts/batch_inference/infer.py`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`

语义：

- 当 `query_visibility_gate_mode=all_future_v1` 时
- 若 query seed 的 query depth `< near_depth_exempt_threshold_m`
- 则该 query 不再受“所有 future frame 都必须可见”的硬约束

这条豁免只作用于 `qvis`，不影响：

- `fixed-view depth gate`
- 后验 `traj_uvd` gate

## 复现实验命令

### 1. `qvis off + fixuvdv1 + trajuvdv1`

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \
/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python \
scripts/batch_inference/infer.py \
  --video_path /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/rgb/stereo_left \
  --depth_path /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/depth/stereo_left \
  --external_geom_npz /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/geom/geom_stereo_left_official_w2c.npz \
  --depth_pose_method external \
  --external_extr_mode w2c \
  --camera_name stereo_left \
  --checkpoint /DATA/disk2/wangchen/projects/TraceForge_Reinforcement/checkpoints/tapip3d_final.pth \
  --out_dir /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/trajectory_dense_none_grid80_notrim_qvisoff_fixuvdv1_trajuvdv1 \
  --scene_storage_mode source_ref \
  --fps 1 \
  --max_num_frames 512 \
  --frame_drop_rate 1 \
  --future_len 32 \
  --grid_size 80 \
  --query_sampler_mode grid \
  --support_grid_ratio 0.0 \
  --grid_border_trim_left 0 \
  --grid_border_trim_right 0 \
  --grid_border_trim_top 0 \
  --grid_border_trim_bottom 0 \
  --filter_level none \
  --traj_filter_profile external \
  --query_prefilter_mode off \
  --query_visibility_gate_mode off \
  --query_fixed_view_depth_gate_mode first_frame_uvd_v1 \
  --traj_uvd_gate_mode delta_uv_depth_v1 \
  --tracker_precision_mode fp32 \
  --num_iters 3 \
  --device cuda:0
```

### 2. `qvis near-exempt 0.8m + fixuvdv1 + trajuvdv1`

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 \
/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python \
scripts/batch_inference/infer.py \
  --video_path /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/rgb/stereo_left \
  --depth_path /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/depth/stereo_left \
  --external_geom_npz /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/geom/geom_stereo_left_official_w2c.npz \
  --depth_pose_method external \
  --external_extr_mode w2c \
  --camera_name stereo_left \
  --checkpoint /DATA/disk2/wangchen/projects/TraceForge_Reinforcement/checkpoints/tapip3d_final.pth \
  --out_dir /DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/trajectory_dense_none_grid80_notrim_qvisv1_nearexempt08_fixuvdv1_trajuvdv1 \
  --scene_storage_mode source_ref \
  --fps 1 \
  --max_num_frames 512 \
  --frame_drop_rate 1 \
  --future_len 32 \
  --grid_size 80 \
  --query_sampler_mode grid \
  --support_grid_ratio 0.0 \
  --grid_border_trim_left 0 \
  --grid_border_trim_right 0 \
  --grid_border_trim_top 0 \
  --grid_border_trim_bottom 0 \
  --filter_level none \
  --traj_filter_profile external \
  --query_prefilter_mode off \
  --query_visibility_gate_mode all_future_v1 \
  --query_visibility_gate_near_depth_exempt_threshold_m 0.8 \
  --query_fixed_view_depth_gate_mode first_frame_uvd_v1 \
  --traj_uvd_gate_mode delta_uv_depth_v1 \
  --tracker_precision_mode fp32 \
  --num_iters 3 \
  --device cuda:0
```

## 当前保留产物

- 原过滤版：
  - `data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/trajectory_dense_none_grid80_notrim_qvisv1_fixuvdv1_trajuvdv1`
- `qvis off`：
  - `data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/trajectory_dense_none_grid80_notrim_qvisoff_fixuvdv1_trajuvdv1`
- `qvis near-exempt 0.8m`：
  - `data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00190_officialprep/trajectory_dense_none_grid80_notrim_qvisv1_nearexempt08_fixuvdv1_trajuvdv1`

## 回归

本轮仅补了 CLI surface 覆盖，未改维护态默认值。

执行：

```bash
PYTHONPATH=. /DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python \
  -m unittest \
  scripts.batch_inference.test_infer_cli_surface \
  scripts.batch_inference.test_press_one_button_demo_utils
```

结果：

- `Ran 40 tests`
- `OK`
