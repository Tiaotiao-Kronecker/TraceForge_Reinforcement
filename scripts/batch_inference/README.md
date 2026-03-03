# 批量推理脚本和文档

本目录包含批量推理相关的所有脚本和文档。

## 脚本文件

### 核心脚本

- **batch_infer.py** - 批量推理主脚本
- **infer.py** - 单个轨迹推理脚本（由batch_infer.py调用）

### 测试和验证脚本

- **stress_test_batch_inference.py** - 压力测试脚本
- **run_large_scale_stress_test.sh** - 大规模压力测试便捷脚本
- **check_batch_inference_results.sh** - 检查批量推理结果脚本
- **analyze_batch_failures.py** - 分析批量推理失败原因
- **verify_model_cache.py** - 验证模型缓存完整性
- **test_model_sharing.py** - 测试模型共享机制

## 文档

- **BATCH_INFERENCE_GUIDE.md** - 批量推理完整指南

## 快速开始

```bash
# 运行批量推理
python batch_infer.py \
    --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
    --out_dir ./output_bridge_depth_grid80 \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5 \
    --gpu_id 0,1,2,3,4,5 \
    --max_workers 6 \
    --grid_size 80
```

详细说明请参考 [BATCH_INFERENCE_GUIDE.md](BATCH_INFERENCE_GUIDE.md)

