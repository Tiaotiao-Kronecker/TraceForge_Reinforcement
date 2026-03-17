# 批量推理指南

本文档只覆盖当前推荐的批量推理用法。

## 前提

- 在仓库根目录运行
- 已安装 `traceforge` 环境
- 已准备 checkpoint

## 通用批量推理

```bash
python scripts/batch_inference/infer.py \
  --video_path <input_dir> \
  --out_dir <output_dir> \
  --batch_process \
  --skip_existing \
  --frame_drop_rate 5 \
  --scan_depth 2 \
  --grid_size 20
```

关键点：

- 默认输出 layout 为 `v2`
- `--output_layout legacy` 仅用于兼容旧工具
- `future_len` 控制每个 query frame 的跟踪窗口
- `grid_size` 控制 query keypoint 密度

## 多 GPU 批处理

```bash
python scripts/batch_inference/batch_infer.py \
  --base_path <dataset_root> \
  --out_dir <output_dir> \
  --gpu_id 0,1,2,3 \
  --skip_existing \
  --frame_drop_rate 5 \
  --grid_size 30
```

适用场景：

- 传统 case 目录批量处理
- 多 GPU 轮询分发
- 需要快速做 smoke run 或大规模批量

## Press-One-Button Demo

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --camera_names varied_camera_1,varied_camera_2,varied_camera_3 \
  --depth_pose_method external \
  --external_geom_name trajectory_valid.h5 \
  --frame_drop_rate 15 \
  --future_len 32 \
  --grid_size 80 \
  --filter_level standard \
  --traj_filter_profile auto
```

补充说明：

- `auto` 会把 wrist-like camera 映射到 `wrist`
- `external_manipulator`、`external_manipulator_v2`、`wrist_manipulator`
  需要显式指定
- 推荐直接在 episode 下写 `trajectory/<camera_name>/...`

## 检查与回归

```bash
python scripts/batch_inference/test_inference_output_shapes.py
python scripts/batch_inference/verify_pointcloud.py
python scripts/batch_inference/verify_traj_valid_mask.py
```

## 不在本文档范围内的内容

- 旧的 SSL/conda 并发排障长记录
- 一次性调查命令
- 已经归档的历史实验流程

这些内容如果仍需要追溯，请看 `docs/history/` 和
`scripts/archived/investigations/`。
