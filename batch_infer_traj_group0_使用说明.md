# batch_infer_traj_group0.py 使用说明

## 📋 功能概述

`batch_infer_traj_group0.py` 是一个批量推理脚本，用于自动处理 `traj_group0` 目录下的所有 `trajX` 子目录。脚本会自动：

1. **自动检测数据格式**：适配两种不同的数据存放格式
   - **格式1**：`trajX/cam0/images0` + `trajX/depth_images0`（如 traj0, traj1）
   - **格式2**：`trajX/images0` + `trajX/depth_images0`（如 traj2, traj3, ...）

2. **批量处理**：按顺序处理所有有效的 traj 目录

3. **错误处理**：支持跳过已存在的输出，支持中断后继续

## 📝 参数说明

### 必需参数

- `--base_path` (必需)
  - **说明**：`traj_group0` 的基础路径
  - **示例**：`/home/user/dataset/bridgeV2/datacol2_toykitchen1/pnp_push_sweep/01/2023-07-03_14-41-20/raw/traj_group0`

### 可选参数

- `--out_dir` (可选，默认：`./output_traj_group0`)
  - **说明**：输出目录，所有 traj 的推理结果将保存在此目录下
  - **示例**：`./output_traj_group0` 或 `/path/to/output`

- `--use_all_trajectories` (可选，标志参数)
  - **说明**：使用所有轨迹（传递给 `infer.py` 的 `--use_all_trajectories` 参数）
  - **用法**：添加此参数即启用，不添加则不启用

- `--skip_existing` (可选，标志参数)
  - **说明**：如果输出目录已存在，则跳过该 traj 的处理
  - **用法**：添加此参数即启用，不添加则不启用
  - **推荐**：在重新运行或中断后继续时使用

- `--frame_drop_rate` (可选，默认：`5`)
  - **说明**：帧采样率，传递给 `infer.py`
  - **示例**：`5` 表示每 5 帧采样一次（即采样第 0, 5, 10, 15, ... 帧）

- `--max_trajs` (可选，默认：`None`)
  - **说明**：最大处理 traj 数量，用于测试或限制处理范围
  - **示例**：`--max_trajs 3` 表示只处理前 3 个 traj
  - **推荐**：首次运行时使用，先测试几个 traj 确认无误后再处理全部

## 🚀 使用示例

### 示例 1：基本用法（处理所有 traj）

```bash
conda run -n traceforge python batch_infer_traj_group0.py \
    --base_path /home/user/dataset/bridgeV2/datacol2_toykitchen1/pnp_push_sweep/01/2023-07-03_14-41-20/raw/traj_group0 \
    --out_dir ./output_traj_group0 \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5
```

### 示例 2：测试模式（只处理前 3 个 traj）

```bash
conda run -n traceforge python batch_infer_traj_group0.py \
    --base_path /home/user/dataset/bridgeV2/datacol2_toykitchen1/pnp_push_sweep/01/2023-07-03_14-41-20/raw/traj_group0 \
    --out_dir ./output_traj_group0_test \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5 \
    --max_trajs 3
```

### 示例 3：不使用 `--use_all_trajectories`

```bash
conda run -n traceforge python batch_infer_traj_group0.py \
    --base_path /home/user/dataset/bridgeV2/datacol2_toykitchen1/pnp_push_sweep/01/2023-07-03_14-41-20/raw/traj_group0 \
    --out_dir ./output_traj_group0 \
    --skip_existing \
    --frame_drop_rate 5
```

### 示例 4：自定义输出目录和帧采样率

```bash
conda run -n traceforge python batch_infer_traj_group0.py \
    --base_path /home/user/dataset/bridgeV2/datacol2_toykitchen1/pnp_push_sweep/01/2023-07-03_14-41-20/raw/traj_group0 \
    --out_dir /home/user/projects/TraceForge/outputs_batch \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 10
```

## 📊 输出结构

脚本会在 `--out_dir` 指定的目录下为每个 traj 创建对应的输出目录：

```
output_traj_group0/
├── traj0/
│   └── images0/
│       ├── images/
│       ├── depth/
│       └── samples/
├── traj1/
│   └── images0/
│       ├── images/
│       ├── depth/
│       └── samples/
├── traj2/
│   └── images0/
│       ├── images/
│       ├── depth/
│       └── samples/
└── ...
```

每个 `trajX/images0/` 目录下的结构与 `infer.py` 的单个推理输出结构相同。

## ⚠️ 注意事项

1. **环境要求**：
   - 必须在 `traceforge` conda 环境中运行
   - 脚本会自动使用 `conda run -n traceforge` 调用 `infer.py`

2. **数据格式要求**：
   - 每个 `trajX` 目录必须包含 `depth_images0` 文件夹
   - 每个 `trajX` 目录必须包含 `cam0/images0` 或 `images0` 文件夹（二选一）
   - RGB 图像和深度图像必须存在且为 `.png`, `.jpg`, 或 `.jpeg` 格式

3. **中断处理**：
   - 使用 `Ctrl+C` 中断后，已处理的 traj 结果会保留
   - 重新运行时使用 `--skip_existing` 可以跳过已处理的 traj

4. **性能建议**：
   - 首次运行建议使用 `--max_trajs 3` 先测试几个 traj
   - 确认无误后再处理全部 traj
   - 每个 traj 的处理时间取决于视频长度和帧数

5. **存储空间**：
   - 确保 `--out_dir` 指定的目录有足够的存储空间
   - 每个 traj 的输出大小约为几十到几百 MB（取决于视频长度）

## 🔍 运行输出示例

```
================================================================================
批量推理 traj_group0
================================================================================

📁 基础路径: /home/user/dataset/bridgeV2/.../traj_group0
📊 找到 25 个有效的traj目录

================================================================================
[1/25] 处理 traj0
================================================================================
  Video: /home/user/dataset/.../traj0/cam0/images0
  Depth: /home/user/dataset/.../traj0/depth_images0
✅ traj0 处理完成

================================================================================
[2/25] 处理 traj1
================================================================================
  Video: /home/user/dataset/.../traj1/cam0/images0
  Depth: /home/user/dataset/.../traj1/depth_images0
✅ traj1 处理完成

...

================================================================================
批量推理完成
================================================================================
  成功: 25/25
  失败: 0/25
  输出目录: ./output_traj_group0
```

## 🐛 常见问题

### Q1: 为什么只找到 2 个有效的 traj？

**A**: 检查数据路径是否正确，确保：
- `--base_path` 指向正确的 `traj_group0` 目录
- 每个 `trajX` 目录下都有 `depth_images0` 和 `images0`（或 `cam0/images0`）
- 图像文件格式为 `.png`, `.jpg`, 或 `.jpeg`

### Q2: 如何处理中断后的继续运行？

**A**: 使用 `--skip_existing` 参数，脚本会自动跳过已存在的输出目录。

### Q3: 如何只处理部分 traj？

**A**: 使用 `--max_trajs N` 参数限制处理数量，例如 `--max_trajs 10` 只处理前 10 个 traj。

### Q4: 输出目录结构是什么？

**A**: 每个 traj 的输出保存在 `{out_dir}/{traj_name}/images0/` 下，包含 `images/`, `depth/`, `samples/` 三个子目录。

