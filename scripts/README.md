# 脚本目录

本目录包含项目中的所有脚本，按功能分类组织。

## 目录结构

```
scripts/
├── README.md                    # 本文件
├── batch_inference/              # 批量推理相关脚本和文档
│   ├── README.md
│   ├── batch_infer.py
│   ├── infer.py
│   ├── stress_test_batch_inference.py
│   ├── run_large_scale_stress_test.sh
│   ├── check_batch_inference_results.sh
│   ├── analyze_batch_failures.py
│   ├── verify_model_cache.py
│   ├── test_model_sharing.py
│   └── BATCH_INFERENCE_GUIDE.md
├── data_analysis/                # 数据分析相关脚本和文档
│   ├── README.md
│   ├── analyze_action_format.py
│   ├── analyze_dataset_structure.py
│   ├── analyze_rotation_representation.py
│   ├── analyze_transform_relationship.py
│   ├── check_action_format.py
│   ├── check_action_info.py
│   ├── verify_transform_relationship.py
│   └── action_data_format_analysis.md
├── visualization/                # 可视化相关脚本和文档
│   ├── README.md
│   ├── visualize_single_image.py
│   └── visualization_features.md
└── archived/                     # 归档脚本（已完成或不再使用）
    ├── find_widowx_urdf.py
    └── check_agent_data_for_urdf.py
```

## 核心脚本（保留在根目录）

以下脚本保留在项目根目录，因为它们是核心功能或环境设置：

- **setup_env.sh** - 环境设置脚本（应在根目录）

## 使用说明

每个子目录都有独立的README.md文件，说明该目录下的脚本和文档。

## 快速链接

- [批量推理脚本](batch_inference/)
- [数据分析脚本](data_analysis/)
- [可视化脚本](visualization/)

