# TraceForge 数据集采样

从大量 TraceForge 推理结果中采样指定数量的 case，用于训练、评估或数据子集构建。

## 使用方式

### 1. 仅生成采样 ID 列表

将 1000 个采样 case 的 ID 保存到文件：

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
    --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
    --n_sample 1000 \
    --output_list sampled_1000.txt
```

输出示例：`sampled_1000.txt` 每行一个 case ID（如 `00000`、`00123`）。

### 2. 复制采样结果到新目录

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
    --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
    --n_sample 1000 \
    --output_dir /data2/dataset/output_bridge_v2_sampled_1000 \
    --output_list sampled_1000.txt \
    --mode copy
```

### 3. 创建软链接（节省磁盘）

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
    --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
    --n_sample 1000 \
    --output_dir /data2/dataset/output_bridge_v2_sampled_1000 \
    --mode symlink
```

### 4. 仅从有效 case 中采样

跳过缺少 `images0/images` 或 `images0/samples` 的失败样本：

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
    --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
    --n_sample 1000 \
    --valid_only \
    --output_list sampled_1000_valid.txt
```

### 5. 均匀采样（按索引等间隔）

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
    --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
    --n_sample 1000 \
    --method uniform \
    --output_list sampled_1000_uniform.txt
```

### 6. 自定义随机种子

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
    --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
    --n_sample 1000 \
    --seed 2024 \
    --output_list sampled_1000_seed2024.txt
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `/data2/dataset/output_bridge_v2_full_grid80` | TraceForge 推理输出根目录 |
| `--n_sample` | 1000 | 采样数量 |
| `--method` | random | 采样方式：`random` 随机，`uniform` 均匀 |
| `--seed` | 42 | 随机种子（仅 `random` 有效） |
| `--valid_only` | False | 仅从有效 case 中采样 |
| `--output_list` | None | 保存采样 ID 列表的文件路径 |
| `--output_dir` | None | 目标目录（配合 `--mode`） |
| `--mode` | copy | 输出模式：`copy` 复制，`symlink` 软链接 |

## 实现逻辑

### 1. Case 枚举

- 扫描 `data_dir` 下所有子目录，每个子目录视为一个 case（如 `00000`、`00001`）
- 若 `--valid_only`，则过滤：要求存在 `images0/images/*.png` 且 `images0/samples/*.npz`

### 2. 采样方式

**随机采样（random）**

- 使用 `random.sample()` 无放回抽样
- 固定 `seed` 可复现结果
- 适用于需要随机子集的场景（如训练/验证划分）

**均匀采样（uniform）**

- 按索引等间隔选取：`indices = [i * (N-1) / (n-1) for i in range(n)]`
- 不依赖随机数，结果确定
- 适用于希望覆盖全量 ID 区间的场景

### 3. 输出

- **ID 列表**：`--output_list` 指定路径，每行一个 case ID
- **复制/链接**：`--output_dir` + `--mode`，将采样 case 复制或软链接到目标目录

### 4. 数据目录结构

采样单位是 case 目录，预期结构：

```
output_bridge_v2_full_grid80/
├── 00000/
│   ├── images0/          # TraceForge 主视角
│   │   ├── images/
│   │   ├── depth/
│   │   ├── samples/
│   │   └── images0.npz
│   ├── images1/
│   ├── images2/
│   ├── lang.txt
│   ├── obs_dict.pkl
│   └── policy_out.pkl
├── 00001/
└── ...
```

采样时复制/链接整个 case 目录，保留全部子目录与文件。
