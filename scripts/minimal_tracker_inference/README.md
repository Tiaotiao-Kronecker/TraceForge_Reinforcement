# Minimal Tracker Inference

这个目录提供一个尽量独立的 3D tracker 推理环境，目标是把下面几件事从
`scripts/batch_inference/infer.py` 那个大脚本里拆出来：

- 只跑底层 tracker，不走 trajectory filtering
- 只保留最小输入：`video/depths/intrinsics/extrinsics/query_points`
- 方便直接比较 `fp32` 和 `bf16`
- 方便做 FLOPs / roofline / H100 理论速度估算
- 方便在真实 episode/camera 上直接做 `fp32 / autocast_bf16 / deep_bf16` A/B 回归

## 输入格式

`run_minimal_tracker_inference.py` 接受一个简单的 `.npz`：

- `video`
  - `[T, H, W, 3]` 或 `[T, 3, H, W]`
  - `uint8` 或 `[0, 1]` 的 `float32`
- `depths`
  - `[T, H, W]`
  - 单位保持和主推理路径一致，通常是米
- `intrinsics`
  - `[T, 3, 3]`
- `extrinsics`
  - `[T, 4, 4]`
  - 按当前 maintained 路径，默认视为 `w2c`
- `query_points_world`
  - 可选
  - `[N, 4]`，列顺序是 `[t, x, y, z]`
  - 如果不提供，脚本会用 `query_grid_size` 在 `query_frame` 上从 depth 反投影一组 world-space query

也可以直接用 `--synthetic` 造一个简单 case 做烟雾测试。

## 用法

FP32：

```bash
/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python \
  scripts/minimal_tracker_inference/run_minimal_tracker_inference.py \
  --checkpoint checkpoints/tapip3d_final.pth \
  --case_npz /path/to/minimal_case.npz \
  --precision_modes fp32 \
  --profile_flops
```

FP32 + BF16 对比：

```bash
/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python \
  scripts/minimal_tracker_inference/run_minimal_tracker_inference.py \
  --checkpoint checkpoints/tapip3d_final.pth \
  --case_npz /path/to/minimal_case.npz \
  --precision_modes fp32,deep_bf16 \
  --support_grid_size 0 \
  --num_iters 3 \
  --profile_flops \
  --output_json /tmp/minimal_tracker_summary.json
```

真实 case A/B：

```bash
/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python \
  scripts/minimal_tracker_inference/run_real_case_precision_ab.py \
  --episode_dir /DATA/disk0/shared/press_one_button_demo_v5_test/episode_00005_blue \
  --camera_name varied_camera_1 \
  --checkpoint ../TraceForge_Reinforcement/checkpoints/tapip3d_final.pth \
  --device cuda:0 \
  --precision_modes fp32,autocast_bf16,deep_bf16 \
  --max_num_frames 32 \
  --query_grid_size 80 \
  --support_grid_size 0 \
  --query_frames auto3 \
  --num_iters 3 \
  --warmup_runs 1 \
  --output_json /tmp/traceforge_real_case_ab.json
```

## 输出说明

脚本会输出：

- `symbolic_complexity`
  - 按当前 `PointTracker3D.streaming_forward()` 的窗口调度估算
  - 包括 window 数、query 数、重复 unprojection 点数、迭代状态更新数
- `runs`
  - 每种精度模式的最佳 wall time
  - `profile_stats`
  - profiler 统计到的 FLOPs
  - H100 SXM/NVL 上的 roofline 下界时间

注意：

- profiler 的 FLOPs 只覆盖 PyTorch 能识别的算子，通常是下界，不是完整精确值
- `autocast_bf16` 和 `deep_bf16` 都不会改变算法复杂度，只改变 kernel 选择、参数/激活 dtype 和可达到的吞吐上限
- `bf16` 这个旧写法现在只作为 `autocast_bf16` 的兼容别名保留
- `--support_grid_size` 默认是 `0`，用于和当前 maintained 路径一样先关闭外层 support points
- 当前底层 tracker 仍有不少显式 `float32` 路径，所以这个脚本更适合回答
  “现有代码在最小环境里到底吃掉了多少时间”和“理论上还有多少空间”
