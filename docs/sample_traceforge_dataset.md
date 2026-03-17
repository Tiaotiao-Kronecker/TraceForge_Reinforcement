# TraceForge 数据集采样

本文档对应 `scripts/data_analysis/sample_traceforge_dataset.py`。

## 当前适用范围

这个脚本当前按 **legacy case 目录** 工作，要求每个 case 至少包含：

```text
<case_dir>/
└── images0/
    ├── images/
    └── samples/
```

它不会解析 press-one-button demo 当前默认的 `v2` scene cache 布局。

## 用法

### 仅输出采样列表

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
  --data_dir <legacy_traceforge_root> \
  --n_sample 1000 \
  --output_list sampled_1000.txt
```

### 复制采样结果

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
  --data_dir <legacy_traceforge_root> \
  --n_sample 1000 \
  --output_dir <target_dir> \
  --output_list sampled_1000.txt \
  --mode copy
```

### 创建软链接

```bash
python scripts/data_analysis/sample_traceforge_dataset.py \
  --data_dir <legacy_traceforge_root> \
  --n_sample 1000 \
  --output_dir <target_dir> \
  --mode symlink
```

## 参数

| 参数 | 说明 |
|------|------|
| `--data_dir` | legacy TraceForge 输出根目录 |
| `--n_sample` | 采样数量 |
| `--method` | `random` 或 `uniform` |
| `--seed` | 随机采样种子 |
| `--valid_only` | 只采样同时含 `images0/images` 和 `images0/samples` 的 case |
| `--output_list` | 保存采样 ID 列表 |
| `--output_dir` | 复制或软链接输出目录 |
| `--mode` | `copy` 或 `symlink` |

## 备注

- 如果后续需要面向 `v2` layout 的采样工具，应单独实现，不要直接把本文档当成当前通用输出结构说明。
