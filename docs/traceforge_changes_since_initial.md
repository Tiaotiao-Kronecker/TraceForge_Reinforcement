# TraceForge 从初版到最新版的主要改动

本文根据当前仓库 Git 历史整理 TraceForge 从初版到最新版的变化。目标不是逐行列出每个提交，而是说明最新维护版相对最初可运行版本在入口、数据输入、预处理、三维轨迹生成、输出协议、批处理、过滤、可视化和工程维护上的实质变化。

## 1. 对比基线

Git 历史里有两个“最初”概念：

| 基线 | 提交 | 日期 | 状态 |
| --- | --- | --- | --- |
| 仓库最早提交 | `ebaacca` | 2025-11-30 | 只有极简 README，没有可运行推理代码。 |
| 最早完整代码提交 | `73426b4` | 2025-12-19 | 第一次包含完整 TraceForge 推理、模型、检查器、可视化和说明。本文把它作为“初版 TraceForge”的功能基线。 |
| 当前最新版 | `642cff5` | 2026-04-21 | 当前 `main` 与 `origin/main` 一致，提交信息为 `Add multi-GPU batching for xperience_raw`。 |

从最早完整代码提交 `73426b4` 到最新版 `642cff5`，Git 统计显示：

```text
226 files changed, 56835 insertions(+), 2707 deletions(-)
```

也就是说，最新版不是在初版上做少量参数调整，而是把原来的单入口推理程序扩展成了面向多数据源、大规模批处理、标准化输出和可验证轨迹质量的维护态数据生成系统。

## 2. 总体变化概览

| 方面 | 初版 TraceForge | 最新维护版 |
| --- | --- | --- |
| 推理入口 | 仓库根目录的 `infer.py`，同时承担单视频、批处理、保存和部分兼容逻辑。 | 维护入口迁移到 `scripts/batch_inference/`；统一批处理入口是 `batch_infer.py --dataset_adapter <sim_file_layout|xperience_raw>`，通用单相机入口是 `scripts/batch_inference/infer.py`。 |
| 数据输入 | 主要面向普通视频目录，按 `--video_path` 扫描输入。 | 明确支持外部深度图、外部相机几何、DROID 数据、仿真 / button episode 布局和原始 Xperience 数据。 |
| 深度和相机几何 | 初版仓库包含单目深度、位姿相关代码，也支持旧式推理路径。 | 当前维护态是 external-only：深度图和相机内外参来自输入数据，不在主流程里用单目深度或位姿模型替代。 |
| 查询帧采样 | 主要用 `frame_drop_rate` 表示每隔多少帧生成一个查询样本。 | 仿真 / button 批处理使用 episode 级共享查询帧计划，按真实帧率每秒采样若干查询帧，并强制保留可用的第 0 帧。 |
| 输出结构 | 默认 legacy 布局：`images/`、`depth/`、`samples/` 和一个整视频 `.npz` 文件。 | 默认 `v2` 布局：`scene_meta.json` 加 `samples/*.npz`；支持 `source_ref`、`adapter_ref` 和 `cache` 三种场景数据保存方式。 |
| 轨迹字段 | 旧样本主要保存轨迹、图像、深度等基础信息。 | 样本保存查询相机坐标下的三维轨迹、查询点、真实帧索引、有效时间步、轨迹级过滤结果、时域监督掩码和过滤诊断信息。 |
| 轨迹质量控制 | 初版没有当前完整的维护态过滤协议。 | 默认使用 `external` 轨迹过滤：基础几何、查询帧深度质量、时序深度一致性和深度波动约束共同决定保留轨迹。 |
| 批处理能力 | 基础批量扫描和跳过已有输出。 | 支持多 GPU 动态调度、共享 GPU worker、显存空闲门控、失败检查、失败重跑、遥测记录和 Xperience 原始数据多 GPU 分片。 |
| 可视化 | 单张图上的三维轨迹查看和基础结果检查器。 | 增加单样本查看、三维轨迹动画、四维重建浏览、baseline/variant 对比、PLY 导出、轨迹 GIF、查询点三联图和 episode 级验证。 |
| 工程组织 | 大量逻辑集中在根目录脚本和模型目录。 | 脚本按 `batch_inference`、`visualization`、`data_analysis`、`archived` 分层；历史调查和过期路径集中归档。 |

## 3. 入口和架构改动

### 3.1 根目录单入口变成分层维护入口

初版 README 推荐的主要命令是：

```bash
python infer.py \
    --video_path <input_video_directory> \
    --out_dir <output_directory> \
    --batch_process \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5 \
    --scan_depth 2
```

最新版已经删除根目录 `infer.py`，维护态推理入口改为：

```bash
python scripts/batch_inference/infer.py \
  --video_path <RGB图像目录或视频文件> \
  --depth_path <深度图目录> \
  --external_geom_npz <相机几何H5或NPZ文件> \
  --camera_name <相机名> \
  --out_dir <输出目录> \
  --checkpoint ./checkpoints/tapip3d_final.pth \
  --skip_existing
```

对大规模数据集，最新版推荐统一批处理入口：

```bash
python scripts/batch_inference/batch_infer.py \
  --dataset_adapter sim_file_layout \
  --base_path <数据集根目录> \
  --camera_names <相机名1,相机名2,...> \
  --gpu_id 0,1,2,3 \
  --skip_existing
```

这次架构变化的核心是：单个脚本不再直接理解所有数据集细节，而是由“数据适配器”负责把不同数据集整理成统一的 RGB 图像、深度图、相机内参、相机外参和时间窗口，再交给同一个三维轨迹生成流程。

### 3.2 统一数据适配器

最新版新增并维护两个主要适配器：

| 适配器 | 面向数据 | 当前作用 |
| --- | --- | --- |
| `sim_file_layout` | 仿真 / button episode 目录 | 读取 episode 下的多相机 RGB 图像、深度图和 `trajectory_valid.h5` 相机几何，生成共享查询帧计划，并按相机输出三维轨迹样本。 |
| `xperience_raw` | 原始 Xperience 数据 | 直接读取 `annotation.hdf5` 和 `stereo_left.mp4`，不要求提前抽帧或转换成 legacy 目录。 |

此外，仓库还保留了 DROID external-only 批处理入口。BridgeV2 / VGGT 相关入口已经从维护态工作流中退休，只保留兼容或占位说明。

## 4. 数据输入支持的变化

### 4.1 初版输入

初版主要围绕视频目录工作。用户给出视频目录和输出目录，程序扫描子目录，抽取图像、深度、样本，并保存到每个视频对应的输出目录。

初版默认输出结构如下：

```text
<输出目录>/
└── <视频名>/
    ├── images/
    ├── depth/
    ├── samples/
    └── <视频名>.npz
```

### 4.2 最新版输入

最新版把输入契约拆得更明确：三维轨迹生成必须同时拿到逐帧 RGB 图像、逐帧深度图、逐帧相机内参和逐帧相机外参。

对仿真 / button 数据，一个 episode 通常是：

```text
<数据集根目录>/
└── episode_00001/
    ├── trajectory_valid.h5
    ├── rgb/
    │   └── <相机名>/
    ├── depth/
    │   └── <相机名>/
    └── lang.txt（可选）
```

对 Xperience 原始数据，一个 episode 通常是：

```text
<Xperience数据集根目录>/
└── <uuid>/
    └── ep7/
        ├── annotation.hdf5
        └── stereo_left.mp4
```

其中 `annotation.hdf5` 提供深度图、深度置信度、相机内参和相机位姿；`stereo_left.mp4` 提供 RGB 视频。最新版会按时间窗口读取这些内容，并保存标准 TraceForge 轨迹样本。

## 5. 深度和相机几何流程的变化

初版仓库包含 MoGE、Depth Anything、ZoeDepth、DepthPro、SpatialTrackerV2 等代码树或相关工具，早期流程对内部估计深度和位姿的兼容路径更多。

最新版维护态明确收敛到 external-only：

1. 深度图来自输入数据。
2. 相机内参来自输入数据。
3. 相机外参来自输入数据。
4. 推理主流程不调用单目深度模型来预测深度。
5. 推理主流程不调用位姿估计模型来替代输入相机几何。

这样做的实际影响是：

- 三维轨迹的几何来源更可追溯。
- 大规模批处理可以复用已有数据集的深度和相机几何。
- 输出可以只保存源数据引用，避免重复缓存整段 RGB 图像和深度图。

## 6. 查询帧采样和时间轴处理的变化

### 6.1 初版采样方式

初版主要使用 `frame_drop_rate` 控制查询帧：例如每隔 5 帧生成一个查询样本。

这种方式简单，但多相机 episode 里不同相机之间不一定自然共享同一批查询帧，也不直接表达“每秒采样多少个关键帧”。

### 6.2 最新版共享查询帧计划

最新版对仿真 / button 多相机数据新增共享查询帧计划：

1. 读取 episode 的真实帧率。
2. 按每秒最少和最多查询帧数量生成候选查询帧。
3. 多个相机共用同一份 episode 级查询帧计划。
4. 如果源数据第 0 帧通过步长、最大帧数和尾段过滤，则强制纳入查询帧计划。
5. 如果某个查询帧到加载后视频末尾的剩余帧数过短，则在生成样本前丢弃。

共享查询帧计划会保存到：

```text
<episode输出目录>/_shared/query_frame_schedule_v3_<hash>.json
```

### 6.3 尾段样本处理

最新版对靠近视频结尾但仍保留的样本做固定长度补齐：

- 真实存在的时间步保留原始三维轨迹。
- 不足设定长度的尾部用无穷值补齐。
- 另存有效时间步标记，说明哪些时间步是真实帧，哪些只是补齐。
- 另存真实帧索引，说明样本时间轴对应原始视频里的哪些帧。

这解决了两个问题：

1. 所有样本在文件中有统一时间长度，方便后续训练或批量读取。
2. 后续程序仍能知道哪些时间步是真实观测，避免把补齐部分当成真实轨迹。

## 7. 三维跟踪模型和查询点机制的变化

### 7.1 当前运行时模型

初版和最新版都围绕 TAPIP3D 权重运行三维点跟踪，默认权重仍是：

```text
checkpoints/tapip3d_final.pth
```

最新版文档和代码明确当前运行时主模型是 TAPIP3D `PointTracker3D`。该模型输入 RGB 图像、深度图、相机内参、相机外参和查询三维点，输出每个查询点在后续帧里的三维轨迹和可见性。

### 7.2 SpatialTrackerV2 归档

初版把 SpatialTrackerV2 代码树放在 `models/SpaTrackV2/` 下。最新版已经把这部分移动到：

```text
third_party/archived/spatrackv2/
```

归档说明明确：SpatialTrackerV2 只作为历史参考或独立 benchmark 代码，不属于当前维护态推理、批处理或训练路径。当前 TraceForge 运行时不需要 SpatialTrackerV2 checkpoint。

### 7.3 查询点能力增强

最新版在查询点机制上增加了多项能力：

- 支持更明确的主查询网格。
- 支持辅助查询点，用于帮助 tracker 在局部三维和图像特征中获得更稳定的匹配证据。
- 支持矩形查询网格，不再只限于正方形网格。
- 支持查询点预过滤，用于在送入 tracker 前先剔除明显低质量查询点。
- 保存时可以把过滤前后的结果映射回完整查询网格，便于可视化和对比。

## 8. 输出协议变化

### 8.1 初版 legacy 输出

初版默认写出完整本地缓存：

```text
<输出目录>/<视频名>/
├── images/
├── depth/
├── samples/
└── <视频名>.npz
```

这种结构直接、便于检查，但在大规模多相机数据集上会重复保存大量 RGB 图像和深度图。

### 8.2 最新版 `v2` 输出

最新版默认输出是：

```text
<相机输出目录>/
├── scene_meta.json
└── samples/
    ├── <样本名>.npz
    └── ...
```

`scene_meta.json` 记录源 RGB 图像、源深度图、源相机几何、真实帧索引、查询帧采样信息和场景保存方式。

每个 `samples/*.npz` 保存一个查询帧对应的轨迹样本，核心内容包括：

| 保存内容 | 含义 |
| --- | --- |
| 查询相机坐标下的三维轨迹 | 每个查询点在样本时间段内的图像横坐标、图像纵坐标和深度。 |
| 查询帧二维点 | 查询帧上实际采样的二维网格点。 |
| 查询帧索引 | 这个样本从哪一帧开始跟踪。 |
| 样本真实帧索引 | 样本时间轴对应原始视频中的哪些帧。 |
| 轨迹保留标记 | 每条轨迹是否通过当前维护态质量过滤。 |
| 时域监督掩码 | 每条轨迹在哪些时间步可作为监督信号。 |
| 有效时间步 | 尾段补齐时，哪些时间步是真实帧。 |
| 过滤诊断信息 | 记录轨迹被保留或拒绝的原因，便于排查过滤策略。 |

### 8.3 场景数据保存方式

最新版的 `v2` 输出支持三种场景数据保存方式：

| 方式 | 含义 | 适用场景 |
| --- | --- | --- |
| `source_ref` | 不复制 RGB 图像、深度图和相机几何，只在 `scene_meta.json` 里记录源路径。 | 仿真 / button、DROID 等已经有明确源文件的数据。 |
| `adapter_ref` | 不复制完整场景数据，只保存结构化数据源描述，由数据适配器在读取时回到原始数据。 | 原始 Xperience 这类由 HDF5 和视频文件组成的数据。 |
| `cache` | 在输出目录写入本地 `scene.h5` 和 `scene_rgb.mp4`。 | 需要输出自包含，或做尺寸变化实验时使用。 |

旧的 legacy 输出仍可通过兼容模式使用，但已经不是推荐默认格式。

## 9. 轨迹过滤和监督信号变化

### 9.1 新增轨迹级保留结果

最新版新增并维护轨迹级保留结果。每条轨迹都会被判断是否适合作为后续训练或可视化使用。

当前维护态默认过滤路径是 `external`。它把三类判断同时作为硬条件：

1. 基础几何是否合理：轨迹要有足够有效帧，深度范围要合理，不能长期跑出图像边界，可见性和深度变化不能异常。
2. 查询帧深度是否可靠：查询点附近的局部深度要足够完整，查询点深度不能明显偏离周围深度。
3. 时序深度是否一致：轨迹投影到后续帧后，要和对应帧深度图在三维上自洽。

只有这三类判断都通过，轨迹才会被标记为保留。

### 9.2 新增时域监督掩码

最新版不只判断整条轨迹是否保留，还会记录每条轨迹在哪些时间步可用作监督。这个信息用于区分：

- 一条轨迹整体是否合格。
- 一条轨迹在某个具体时间步是否有可靠三维监督。
- 尾段样本中哪些时间步来自真实帧，哪些时间步只是固定长度补齐。

### 9.3 保留历史 wrist / manipulator 过滤分支

最新版仍保留 wrist、manipulator、pick-place 等显式过滤分支，用于历史实验、兼容复跑或定向分析。但当前维护态默认已经收敛到 external-only，`auto` 也只是兼容别名，实际解析为 `external`。

## 10. 批处理、性能和稳定性变化

最新版相对初版增加了大量面向大规模数据生成的能力。

### 10.1 多 GPU 和任务调度

最新版批处理支持：

- 通过 `--gpu_id 0,1,2,3` 指定多张 GPU。
- 每张 GPU 启动一个或多个 worker。
- 动态分发 episode / 相机任务。
- 按空闲显存阈值等待 GPU 恢复。
- 遇到可重试的 CUDA 失败后重新排队。
- 对 Xperience 原始数据进行多 GPU 分片处理。

### 10.2 性能优化

Git 历史显示，最新版陆续加入了：

- 深度过滤热点路径优化。
- 深度波动计算向量化。
- manipulator motion 指标向量化。
- 推理和保存阶段的耗时分解。
- 查询点预过滤。
- 相机级 tracker 迭代次数覆盖。
- 支持共享 GPU workers。
- 处理速度 benchmark 和多配置对比工具。

这些改动的目标不是改变最终数据语义，而是让同样的三维轨迹生成流程可以在更多 episode、更长视频和多相机数据上稳定运行。

### 10.3 遥测和分析

最新版新增批处理遥测输出和分析工具，可以记录：

- 每个 episode 和相机的处理耗时。
- 推理阶段、深度过滤阶段、保存阶段的耗时。
- GPU 静态信息和运行时采样。
- 成功、失败、跳过、重试等批处理状态。

这些信息由 `scripts/data_analysis/` 下的分析脚本汇总，用于定位瓶颈和比较不同推理配置。

## 11. 可视化和验证变化

初版主要提供单样本三维轨迹查看和两个基础结果检查器。

最新版可视化能力扩展为：

| 工具 | 用途 |
| --- | --- |
| `scripts/visualization/visualize_single_image.py` | 读取单个样本，并按轨迹保留结果显示三维轨迹。 |
| `scripts/visualization/visualize_3d_keypoint_animation.py` | 按时间播放三维关键点轨迹，可同时显示稠密点云。 |
| `scripts/visualization/visualize_4d_reconstruction.py` | 聚合一个相机输出目录下的多个样本，浏览动态点云、轨迹和相机视锥。 |
| `scripts/visualization/visualize_3d_keypoint_comparison.py` | 对比 baseline 和 variant 两套输出，显示共同保留、仅 baseline 保留、仅 variant 保留的轨迹。 |
| `scripts/visualization/verify_episode_trajectory_outputs.py` | 对 episode 输出做验证，并导出 PLY、验证图或 GIF。 |
| `scripts/visualization/export_ranked_query_triptychs.py` | 导出按查询点差异排序的三联图，辅助比较过滤或参数变化。 |

最新版还新增了统一的 `SceneReader` 读取层，使可视化工具可以读取 `source_ref`、`adapter_ref` 和 `cache` 三类 `v2` 输出，而不是只依赖旧式本地图片和深度目录。

## 12. 文档、测试和工程维护变化

最新版新增了大量当前维护文档和历史归档：

- 当前输出结构说明。
- 当前数据预处理和三维轨迹生成流程说明。
- 当前轨迹过滤逻辑说明。
- 底层三维 tracking 和 refinement 数学说明。
- 深度尺度对齐说明。
- 支持网格行为说明。
- 历史排查、性能实验和验证记录归档。

最新版也新增了大量单元测试和工具测试，覆盖：

- 外参方向处理。
- 查询帧计划。
- 输出 artifact 读取。
- 轨迹过滤逻辑。
- Xperience 原始数据适配器。
- 批处理命令行接口。
- 可视化导出工具。
- 数据分析和 benchmark 工具。

工程组织上，最新版把一次性调查脚本移动到 `scripts/archived/investigations/`，把 SpatialTrackerV2 移动到 `third_party/archived/spatrackv2/`，并在当前 README 和脚本文档中明确说明哪些入口是维护态，哪些只是历史兼容。

## 13. 已退休或降级为兼容路径的内容

最新版相对初版删除或降级了以下内容：

| 内容 | 最新状态 |
| --- | --- |
| 根目录 `infer.py` | 已删除；维护态入口迁移到 `scripts/batch_inference/infer.py`。 |
| 根目录 `visualize_single_image.py` | 已删除；维护态入口迁移到 `scripts/visualization/visualize_single_image.py`。 |
| legacy 默认输出 | 仍可兼容读取或显式写出，但不再是默认推荐格式。 |
| BridgeV2 / VGGT 专用推理入口 | 已从维护态工作流退休。 |
| DROID legacy 输出 | 已移除，DROID 当前走 external-only `v2 + source_ref`。 |
| SpatialTrackerV2 运行时路径 | 已归档到 `third_party/archived/spatrackv2/`，不参与当前 TraceForge 运行时。 |
| 历史一次性调查脚本 | 已集中归档到 `scripts/archived/investigations/`。 |

## 14. 对使用者的实际影响

如果你从初版 TraceForge 迁移到最新版，需要关注以下变化：

1. 输入不再只是“给一个视频目录”。当前维护态需要明确的 RGB 图像、深度图、相机内参和相机外参，或者使用已有的数据适配器从数据集中读取这些内容。
2. 输出不再默认复制整段图像和深度。默认 `v2 + source_ref` 或 `v2 + adapter_ref` 会通过元数据回到源数据读取场景信息。
3. 样本文件更适合训练和质量控制。每个样本不仅有三维轨迹，还有真实帧索引、有效时间步、轨迹保留结果和时域监督信息。
4. 多相机 episode 可以共享同一批查询帧，方便跨相机比较和后续训练对齐。
5. 大规模处理能力明显增强。最新版支持多 GPU 动态调度、失败重试、遥测分析和性能 benchmark。
6. 历史模型和旧入口仍能在仓库中找到线索，但当前维护态应优先使用 `scripts/batch_inference/`、`scripts/visualization/` 和 `docs/` 中列出的入口。

## 15. 推荐阅读顺序

要理解最新版维护态，建议按下面顺序阅读：

1. [`README.md`](../README.md)：当前推荐入口和输出概览。
2. [`docs/data_preprocessing_generation_flow.md`](data_preprocessing_generation_flow.md)：完整数据预处理和三维轨迹生成流程。
3. [`docs/traceforge_output_structure.md`](traceforge_output_structure.md)：当前 `v2` 输出协议。
4. [`docs/maintained_traj_filter_logic.md`](maintained_traj_filter_logic.md)：当前维护态轨迹过滤逻辑。
5. [`docs/traceforge_tracking_math.md`](traceforge_tracking_math.md)：底层三维跟踪和 refinement 机制。
6. [`scripts/batch_inference/README.md`](../scripts/batch_inference/README.md)：批处理入口、适配器和 smoke 验证记录。

