# 脚本目录

本目录按功能划分当前可用脚本，并把历史一次性调查脚本统一收进归档区。

## 当前目录

- `batch_inference/`
  - 推理、批处理、校验和回归测试入口
- `visualization/`
  - 3D 可视化、PLY 导出、episode 验证
- `data_analysis/`
  - 当前仍保留的数据分析和数据集工具
- `archived/`
  - 历史脚本，不作为当前工作流入口

## 推荐入口

- `scripts/batch_inference/infer.py`
- `scripts/batch_inference/batch_infer.py`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`
- `scripts/visualization/visualize_single_image.py`
- `scripts/visualization/verify_episode_trajectory_outputs.py`

## 历史脚本

一次性调查脚本已迁移到：

- `scripts/archived/investigations/2026-03/`

如果某个历史文档引用了旧脚本路径，请以该归档目录中的脚本为准。
