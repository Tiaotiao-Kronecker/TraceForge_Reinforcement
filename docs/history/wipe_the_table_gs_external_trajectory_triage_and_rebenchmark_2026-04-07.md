# wipe_the_table_gs 外部相机轨迹排查与 external-only 复测矩阵

日期：2026-04-07

## 背景

下游同事反馈 `/DATA/disk6/gs_datasets/wipe_the_table_gs` 的外部相机轨迹存在明显错误，需要先找到当时的历史推理输出做可视化排查，再把当前维护态 `external-only` 的复测矩阵落成明确入口。

## 历史推理结果定位

原始数据目录并不直接保存 TraceForge 推理结果。历史输出实际位于：

- `/DATA/disk1/zoyo/mcap/wipe_the_table_gs/<episode>/trajectory/<camera_name>/samples/*.npz`

已确认至少以下 episode 有可直接复用的历史输出：

- `00000`
- `00001`
- `00002`

这些 case 的 `scene_meta.json` 采用 `scene_storage_mode=source_ref`，并引用回：

- `/DATA/disk1/zoyo/mcap/wipe_the_table_gs/<episode>/rgb/<camera_name>`

因此可以直接从历史 `sample npz + source rgb` 生成轻量可视化，而不必重跑完整推理。

## 可视化产物

由于当前节点缺少 `numpy`、`h5py`、`matplotlib`，新增了只依赖标准库加 Pillow 的轻量导出脚本：

- `scripts/visualization/export_lightweight_trajectory_overlays.py`

已生成的外部相机排查产物位于：

- `/DATA/disk3/tmp/external_preview_20260407/wipe_the_table_gs_00000_q20`
- `/DATA/disk3/tmp/external_preview_20260407/wipe_the_table_gs_00001_q14`
- `/DATA/disk3/tmp/external_preview_20260407/wipe_the_table_gs_00002_q20`

每个目录包含：

- `summary.json`
- `episode_overlay_overview.png`
- `<camera_name>/<camera_name>_frameXXXXX_overlay.png`
- `<camera_name>/<camera_name>_frameXXXXX_overlay.gif`

当前选取的排查 query frame 为：

- `00000`: `q=20`
- `00001`: `q=14`
- `00002`: `q=20`

## 观察结论

在 `00000 / 00001 / 00002` 三个 episode 上，三路外部相机都出现了稳定复现的异常模式：

1. `varied_camera_1`
   - 大批轨迹被压成近似竖直平行线。
   - 轨迹整体悬浮在玩偶上方，和真实物体表面运动不对齐。
2. `varied_camera_2`
   - 同样出现竖直塌缩，但位置更集中。
   - 轨迹从筐体附近直接“落”到前景玩偶，明显不像正常局部跟踪。
3. `varied_camera_3`
   - 不表现为竖直塌缩，而是从玩偶两侧向图像下边缘大范围扇形发散。
   - 这更像整体投影几何或 frame/source 对应关系错位，而不是少量 bad tracks。

这些异常在不同 episode 上形态高度一致，更像系统性问题，而不是单个 query frame 的随机漂移。

## external-only 复测矩阵

当前维护态 benchmark 默认口径已经收敛到：

- 相机：`varied_camera_1,varied_camera_2`
- `traj_filter_profile=external`
- `num_iters=5` 作为默认基线

结合 `docs/history/external_only_speed_reassessment_2026-04-07.md`，本轮建议的 external-only 复测矩阵为三步：

### 1. Pilot sweep

目的：

- 快速确认 `num_iters=5/4/3` 在 `wipe_the_table_gs` 上的速度收益和可视化差异

命令：

```bash
python scripts/data_analysis/benchmark_num_iters_manifest.py \
  --manifest scripts/data_analysis/manifests/wipe_the_table_gs_external_only_pilot_20260407.json \
  --camera-names varied_camera_1,varied_camera_2 \
  --num-iters-values 5,4,3 \
  --baseline-num-iters 5 \
  --support-grid-ratio 0 \
  --warmup-runs 1 \
  --benchmark-runs 1 \
  --run-visual-verification
```

### 2. Median3 baseline telemetry

目的：

- 固定 `num_iters=5`
- 观察 `process_total_seconds`
- 观察 `tracker_model_forward_seconds`
- 观察 `prepare_depth_filter_*` 的波动

命令：

```bash
python scripts/data_analysis/benchmark_num_iters_manifest.py \
  --manifest scripts/data_analysis/manifests/wipe_the_table_gs_external_only_median3_20260407.json \
  --camera-names varied_camera_1,varied_camera_2 \
  --num-iters-values 5 \
  --baseline-num-iters 5 \
  --support-grid-ratio 0 \
  --warmup-runs 1 \
  --benchmark-runs 3
```

### 3. Candidate confirmation

触发条件：

- 只有在 pilot sweep 显示 `num_iters=4` 的速度收益明确、且 visual verification 没有新增明显退化时，才继续

命令：

```bash
python scripts/data_analysis/benchmark_num_iters_manifest.py \
  --manifest scripts/data_analysis/manifests/wipe_the_table_gs_external_only_median3_20260407.json \
  --camera-names varied_camera_1,varied_camera_2 \
  --num-iters-values 5,4 \
  --baseline-num-iters 5 \
  --support-grid-ratio 0 \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --run-visual-verification
```

## 当前节点限制

截至 2026-04-07，本节点不具备直接执行上述 benchmark 的前提：

1. `conda env list` 只有 `base`
2. `python3` 缺少：
   - `numpy`
   - `h5py`
   - `matplotlib`
3. `nvidia-smi -L` 无法连通 NVIDIA driver

因此当前节点适合：

- 历史输出定位
- 轻量可视化导出
- manifest / 文档 / 运行入口准备

不适合直接启动 GPU 复测。

## 当前结论

1. `wipe_the_table_gs` 的历史外部相机结果已经定位完成，并已生成可直接查看的 PNG/GIF 排查产物。
2. 三路外部相机都存在稳定异常，但 `varied_camera_1,2` 与 `varied_camera_3` 的失真形态不同，值得后续分别追根因。
3. external-only 的复测矩阵已经收敛成可执行 manifest 和命令入口；真正跑 benchmark 需要切换到具备完整 Python 依赖和 GPU 驱动的节点。
