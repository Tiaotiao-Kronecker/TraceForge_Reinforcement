# TraceForge 数据预处理与生成流程

本文说明当前维护态 TraceForge 如何把机器人操作数据预处理成三维轨迹样本。这里的“预处理和生成”指推理生成数据集产物：读取 RGB 图像、深度图和相机几何，运行三维点跟踪，过滤不可靠轨迹，并保存为训练或可视化可读取的样本文件。

说明原则：

- 正文尽量使用数据本身来描述，例如“RGB 图像序列”“深度图”“相机内参”“相机外参”“三维点云”“三维轨迹”。
- 表格中的英文名称是命令行参数、文件名或落盘字段名，保留它们是为了能和真实输入输出一一核对。
- 当前维护态默认不做弧长重采样或速度重定标。样本时间轴来自加载后的真实帧顺序；尾段不足设定长度时用无穷值补齐，并用有效时间步标记真实前缀。

## 1. 当前维护态总览

当前主流程是：

```text
数据集目录
  -> 发现 episode / 相机 / 时间窗口
  -> 读取 RGB 图像、深度图、相机内参、相机外参
  -> 对齐帧顺序、采样步长和最大帧数
  -> 生成共享查询帧计划
  -> 清理深度图边缘噪声
  -> 在查询帧上采样二维网格点
  -> 用深度图和相机几何把二维点抬升为世界坐标三维点
  -> TAPIP3D 三维点跟踪
  -> 把世界坐标三维轨迹投影到查询帧相机坐标
  -> 轨迹质量过滤
  -> 保存 scene_meta.json 和 samples/*.npz
```

维护态默认入口是统一批处理脚本：

```bash
python scripts/batch_inference/batch_infer.py \
  --dataset_adapter sim_file_layout \
  --base_path <数据集根目录> \
  --camera_names <相机名1,相机名2,...> \
  --checkpoint ./checkpoints/tapip3d_final.pth \
  --gpu_id 0,1,2,3 \
  --skip_existing
```

也可以处理原始 Xperience 数据：

```bash
python scripts/batch_inference/batch_infer.py \
  --dataset_adapter xperience_raw \
  --dataset_root <Xperience 数据集根目录> \
  --episode_glob '<uuid>/ep7' \
  --camera_name stereo_left \
  --checkpoint ./checkpoints/tapip3d_final.pth \
  --out_dir <输出根目录> \
  --skip_existing
```

通用单视频或单相机目录也可以直接使用：

```bash
python scripts/batch_inference/infer.py \
  --video_path <RGB 图像目录或视频文件> \
  --depth_path <深度图目录> \
  --external_geom_npz <相机几何 H5 或 NPZ 文件> \
  --camera_name <H5 中的相机名> \
  --out_dir <输出根目录> \
  --checkpoint ./checkpoints/tapip3d_final.pth \
  --skip_existing
```

## 2. 输入数据契约

### 2.1 仿真 / button 文件布局

这个布局由 `sim_file_layout` 数据适配器读取。一个 episode 至少需要：

```text
<数据集根目录>/
└── episode_00001/
    ├── trajectory_valid.h5
    ├── rgb/
    │   ├── varied_camera_1/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── varied_camera_2/
    ├── depth/
    │   ├── varied_camera_1/
    │   │   ├── 000000.npy 或 000000.png
    │   │   └── ...
    │   └── varied_camera_2/
    └── lang.txt（可选）
```

必需输入含义：

| 输入 | 含义 |
| --- | --- |
| `rgb/<相机名>/` | 某个相机按时间排序的 RGB 图像帧，支持 PNG、JPG、JPEG。 |
| `depth/<相机名>/` | 与 RGB 图像逐帧对应的深度图。NPY 深度默认是米；16-bit PNG 深度按毫米读取并转换成米。 |
| `trajectory_valid.h5` | 每帧相机内参、每帧相机外参，以及 episode 的真实帧率。 |
| `trajectory_valid.h5` 根属性 `fps` | 共享查询帧采样使用的真实 episode 帧率。 |

H5 里相机几何的默认读取路径是：

```text
observation/camera/intrinsics/<相机名>_left
observation/camera/extrinsics/<相机名>_left
```

如果没有 `_left` 后缀，会尝试：

```text
observation/camera/intrinsics/<相机名>
observation/camera/extrinsics/<相机名>
```

### 2.2 原始 Xperience 数据

这个布局由 `xperience_raw` 数据适配器读取。当前维护相机是 `stereo_left`。

```text
<Xperience 数据集根目录>/
└── <uuid>/
    └── ep7/
        ├── annotation.hdf5
        └── stereo_left.mp4
```

必需输入含义：

| 输入 | 含义 |
| --- | --- |
| `stereo_left.mp4` | 原始左目 RGB 视频。读取时按指定帧号解码，不要求先抽帧。 |
| `annotation.hdf5` 中的 `depth/depth` | 与视频对齐的深度图，按米参与后续三维计算。 |
| `annotation.hdf5` 中的 `depth/confidence` | 深度置信度，作为场景附加信息保存。 |
| `annotation.hdf5` 中的 `calibration/cam01/K` | 左目相机内参，原始格式是焦距和主点，读取后转成 3×3 相机内参矩阵。 |
| `annotation.hdf5` 中的 `slam/trans_xyz` 和 `slam/quat_wxyz` | 每帧相机位姿，读取后转成世界到相机的 4×4 外参矩阵。 |
| `caption`（可选） | 任务文字描述，若存在会保存到输出目录的 `lang.txt`。 |

Xperience 会先按原始 episode 切成时间窗口，再对每个窗口独立生成样本。

### 2.3 通用单视频 / 单相机目录

直接使用 `infer.py` 时，需要显式提供三类数据：

| 输入 | 含义 |
| --- | --- |
| RGB 图像目录或视频文件 | 图像目录按文件名中的数字自然排序；视频文件会被整体读取后再按步长采样。 |
| 深度图目录 | 与 RGB 图像同顺序的深度图；NPY 按米读取，16-bit PNG 按毫米转换成米。 |
| 相机几何 H5 或 NPZ | 每帧相机内参和外参。NPZ 必须包含 `intrinsics` 和 `extrinsics` 两个数组。 |

当前维护态强制使用外部深度和外部相机几何；不会调用单目深度估计或位姿估计模型来替代这些输入。

## 3. 所用模型和关键组件

### 3.1 运行时真正使用的模型

| 模型或组件 | 用途 |
| --- | --- |
| TAPIP3D `PointTracker3D` | 主三维点跟踪模型。输入 RGB 图像、深度图、相机内参、相机外参和查询三维点，输出每个查询点在后续帧里的三维轨迹和可见性。 |
| CoTracker 风格 RGB 特征编码器 | 从 RGB 图像中提取用于匹配和跟踪的视觉特征。它是 TAPIP3D 模型内部的一部分。 |
| 四维相关性 / 近邻特征模块 | 在时空点云和图像特征中为每个查询点构建局部匹配证据。它也是 TAPIP3D 模型内部的一部分。 |
| `EfficientUpdateFormer` | TAPIP3D 的迭代更新模块，根据相关性特征、相对运动和可见性估计逐步修正三维坐标。 |
| `pointops2` CUDA 扩展 | 用于点云近邻、分组、采样等高性能运算。 |

默认模型权重路径：

```text
checkpoints/tapip3d_final.pth
```

下载地址在 README 中给出：

```bash
wget -O checkpoints/tapip3d_final.pth \
  https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```

### 3.2 当前不会在维护态主流程中调用的模型

仓库里仍有 MoGE、Depth Anything、ZoeDepth、DepthPro、归档的 SpatialTrackerV2 等代码，但当前维护态推理链路是 external-only：

- 深度图来自输入数据，不由这些模型预测。
- 相机内参和外参来自输入数据，不由这些模型估计。
- 归档的 SpatialTrackerV2 只保留作历史对比，不是当前运行时入口。

### 3.3 可选语言指令生成模型

`text_generation/generate_description.py` 是独立的可选步骤，不参与三维轨迹生成。它会从 episode 的 `images/` 目录选取 5 张图，调用视觉语言模型生成三个短指令，默认 OpenAI 模型名是 `gpt-4o-mini`，也支持 Google Gemini。输出文件默认是：

```text
three_instructions.json
```

## 4. 核心预处理原理

### 4.1 RGB、深度和相机几何必须逐帧对齐

每一帧三维计算都依赖三件事同时正确：

1. 当前帧 RGB 图像提供视觉纹理。
2. 当前帧深度图提供每个像素到相机的距离。
3. 当前帧相机内参和外参提供像素、相机坐标和世界坐标之间的变换。

因此程序会统一按同一个帧顺序、同一个采样步长、同一个最大帧数裁剪 RGB 图像、深度图、相机内参和相机外参。若三者长度不同，只使用共同可用的前缀，避免后续出现图像帧和相机几何错位。

### 4.2 相机外参统一成世界到相机

输入几何可能是世界到相机，也可能是相机到世界。当前程序内部统一使用“世界坐标到相机坐标”的 4×4 外参矩阵。如果输入声明为相机到世界，程序会先求逆。

统一外参方向的原因是：三维点跟踪和深度一致性检查都需要频繁把世界坐标里的点投影回每一帧相机图像。固定一种外参语义可以减少坐标系错误。

### 4.3 深度图单位和清理

深度图进入模型前统一使用米：

- NPY 深度图按米读取。
- 16-bit PNG 深度图按毫米读取，再除以 1000 转成米。
- Xperience 的 HDF5 深度按米参与计算。

在每个查询帧对应的跟踪片段里，程序还会对深度图做一次边缘清理。原理是：

1. 用相机内参把深度图反投影成稠密三维点云。
2. 从三维点云估计表面法线。
3. 同时检测深度突变边缘和法线突变边缘。
4. 对被判定为高风险边缘的深度像素，用最近的非边缘有效深度替换。

这一步的目的不是生成新的真实深度，而是减轻深度边缘处的飞点和射线伪影，让查询点抬升到三维空间时更稳定。轨迹保存和时域一致性检查仍会保留对原始深度的引用和比较。

### 4.4 查询帧计划

查询帧是要在哪些时间点启动一批三维点跟踪。当前批处理不再简单“每隔固定帧数取一帧”，而是优先按真实秒数采样：

1. 先根据加载步长和最大帧数得到候选帧。
2. 去掉离视频结尾太近的候选帧；如果某帧到片段末尾剩余帧数小于 9 帧，就不作为查询帧。
3. 按 episode 的真实帧率把候选帧分到每一秒。
4. 每秒随机但可复现地采样指定数量的查询帧，默认每秒 2 到 3 帧。
5. 如果原始第 0 帧仍在候选集中，则强制加入，保证每个 episode 或窗口都有标准起始样本。

仿真 / button 文件布局会在同一个 episode 内给多个相机共用一份查询帧计划，确保多相机输出在时间上对齐。Xperience 则对每个时间窗口独立生成查询帧计划。

### 4.5 查询点网格和三维点云

每个查询帧上会生成一张均匀二维网格。默认网格边长是 80，也就是每个查询帧最多生成 6400 个主查询点。每个二维网格点会按下面的物理关系抬升到世界坐标：

1. 在查询帧深度图上读取该像素位置的深度。
2. 用相机内参把像素坐标和深度变成相机坐标里的三维点。
3. 用查询帧相机外参的逆变换，把相机坐标三维点变成世界坐标三维点。

TAPIP3D 实际收到的查询不是“二维像素点”，而是“某个查询时刻的世界坐标三维点”。

### 4.6 支撑网格

除了最终要保存的主查询点，程序还可以额外采样一批支撑点。默认支撑网格边长是主网格边长的 0.8 倍。

支撑点的作用是增强 TAPIP3D 在局部三维点云中的匹配上下文。它们会和主查询点一起进入模型，参与完整跟踪，但在模型输出返回到保存流程前会被裁掉，因此最终样本只保存主查询点的轨迹。

### 4.7 三维跟踪为什么能补偿相机运动

模型跟踪发生在由深度图和相机几何构成的三维空间里，而不是只在二维图像上追像素。每帧的相机外参都告诉程序相机本身如何运动，因此同一个真实物体表面点可以在世界坐标里保持连续，再投影回不同相机帧进行检查。

保存时，程序会把世界坐标轨迹投影回“查询帧相机视角”的坐标，得到“横向像素、纵向像素、深度”的三维轨迹。这相当于用查询帧作为该样本的固定参考相机，降低后续训练或可视化对原始相机运动的依赖。

## 5. 程序处理步骤

### 步骤 1：选择数据适配器

统一入口 `batch_infer.py` 只负责分发：

- `sim_file_layout` 读取维护态仿真 / button episode 目录。
- `xperience_raw` 直接读取原始 Xperience episode。

分发后，两类数据都会收敛到同一套核心推理和保存函数。

### 步骤 2：发现待处理任务

仿真 / button 文件布局按下面条件发现 episode：

- episode 目录名以 `episode_` 开头，或目录名是纯数字。
- episode 目录里存在相机几何文件，默认是 `trajectory_valid.h5`。
- 至少一个指定相机同时拥有 RGB 图像目录和深度图目录。

每个“episode + 相机”是一项相机任务。未显式传输出根目录时，结果写回 episode 内部的 `trajectory/` 目录；显式传输出根目录时，结果写到输出根目录下对应 episode 名目录。

Xperience 按 `annotation.hdf5` 发现 episode，再按起止帧和窗口长度切成多个时间窗口。每个“episode 窗口 + 相机”是一项窗口任务。

### 步骤 3：加载 RGB 图像、深度图和相机几何

仿真 / button 输入：

- RGB 图像目录按自然顺序排序，避免 `im_10` 排在 `im_2` 前面。
- 深度图目录使用同样排序规则。
- RGB 图像转换成 0 到 1 范围的浮点张量。
- 深度图转换成米单位的浮点数组。
- 相机内参和外参从 H5 或 NPZ 读取，并按加载步长同步采样。

Xperience 输入：

- 使用 `ffmpeg` 按指定帧号从 `stereo_left.mp4` 读取 RGB 图像。
- 从 `annotation.hdf5` 读取深度图和深度置信度。
- 从标定向量构建相机内参矩阵；如果 RGB 原始分辨率和深度图分辨率不同，会把相机内参缩放到深度图分辨率。
- 从 SLAM 平移和四元数构建每帧世界到相机的外参矩阵。

### 步骤 4：生成查询帧计划

仿真 / button 输入会在 episode 输出目录下写共享计划：

```text
<episode 输出目录>/_shared/query_frame_schedule_v3_<哈希>.json
```

这个文件记录：

- 参与生成计划的相机列表。
- episode 真实帧率。
- 加载步长和最大帧数。
- 各相机共同可用的原始帧数。
- 候选源帧。
- 因为尾部太短而丢弃的源帧。
- 最终选中的查询源帧。

核心推理会读取这个计划，把源帧编号映射成当前加载后的视频内帧编号。这样即使不同批次只处理部分相机，只要共享计划相同，查询帧仍然一致。

### 步骤 5：为每个查询帧切出跟踪片段

对于每个查询帧，程序从该帧开始向后取一个固定长度的片段。维护态仿真批处理默认长度是 32 帧；Xperience 默认长度是 16 帧；直接 `infer.py` 默认是 128 帧。

如果查询帧后面没有足够长的视频，程序最多取到视频结尾。只要片段长度大于 8 帧，仍然可以保存；不足设定长度的时间轴会在保存时补齐。

### 步骤 6：清理片段深度图

每个跟踪片段里的每一帧深度都会经过边缘清理。程序会缓存已经清理过的帧，因为相邻查询帧的跟踪片段可能重叠，重复使用缓存可以减少预处理开销。

### 步骤 7：构造查询三维点

对每个查询帧：

1. 在查询帧上生成主查询二维网格。
2. 读取清理后的查询帧深度。
3. 用相机内参和外参把二维网格点抬升为世界坐标三维点。
4. 可选地生成支撑三维点，与主查询点一起送入模型。

如果启用了查询点预筛选，程序会先去掉明显不可靠的查询点；跟踪结束后再把结果散回完整网格，使落盘样本仍保持完整网格数量。

### 步骤 8：运行 TAPIP3D 三维跟踪

TAPIP3D 处理一个跟踪片段时会做这些事情：

1. 从 RGB 图像提取视觉特征。
2. 用深度图、相机内参和相机外参反投影出每帧稠密三维点云。
3. 用深度分布估计有效深度范围，减少远处异常深度对归一化的影响。
4. 在模型内部把三维坐标归一化到稳定工作尺度。
5. 在重叠时间窗口内迭代跟踪三维点。
6. 每次迭代利用图像特征、三维点云近邻、相对运动和可见性状态修正坐标。
7. 输出每个主查询点在片段内每一帧的世界坐标三维轨迹，以及可见性判断。

维护态默认迭代次数是 3。更高迭代次数通常更稳但更慢。

### 步骤 9：转换成查询帧相机坐标

模型输出是世界坐标三维轨迹。保存前会投影到查询帧相机视角，变成：

```text
横向像素位置、纵向像素位置、查询帧相机坐标系下的深度
```

这就是主样本里保存的三维轨迹格式。它不是程序内部类，也不是二维轨迹；它是以查询帧相机作为固定参考视角的三维轨迹表达。

### 步骤 10：轨迹过滤

当前维护态默认过滤配置是：

```text
filter_level = standard
traj_filter_profile = external
```

默认外部相机轨迹过滤由三类检查共同决定：

1. 基础几何检查  
   轨迹需要有足够多有效帧；深度要在合理范围内；投影不能大幅跑出图像；可见比例要足够；深度变化不能剧烈抖动。

2. 查询帧深度质量检查  
   查询点周围的小深度块需要有足够有效深度；查询点深度不能明显偏离周围深度中位数。

3. 时域深度一致性检查  
   把查询帧相机坐标里的轨迹还原成世界坐标，再投影到片段内每一帧相机，和原始深度图比较。若投影深度与该帧观测深度在绝对 0.05 米或相对 10% 容差内一致，认为该时间步自洽。默认要求一致比例达到 0.95。

程序还会识别深度随时间特别不稳定的像素区域，避免这些高波动区域过度影响时域一致性判断。

显式指定机械臂或 wrist 相关过滤配置时，还会额外根据近景深度、世界坐标运动幅度、二维空间连通区域、pick-place 热力图或延迟接触关系做保留或剔除。但这些不是当前维护态默认路径。

### 步骤 11：保存样本和场景元数据

默认输出协议是 `v2`。仿真 / button 输入默认写 `source_ref`，也就是只在 `scene_meta.json` 中记录源 RGB、源深度和源相机几何路径，不复制完整场景缓存。这样输出更小，读取时再回源数据加载场景。

如果显式使用 `cache`，程序会额外写：

```text
scene.h5
scene_rgb.mp4
```

其中 `scene.h5` 保存完整深度图、相机内参和世界到相机外参；`scene_rgb.mp4` 保存完整 RGB 缓存。

## 6. 完整输出结构

### 6.1 仿真 / button 默认原地输出

未传输出根目录时：

```text
<episode 目录>/
└── trajectory/
    ├── _shared/
    │   └── query_frame_schedule_v3_<哈希>.json
    ├── varied_camera_1/
    │   ├── scene_meta.json
    │   └── samples/
    │       ├── varied_camera_1_0.npz
    │       ├── varied_camera_1_24.npz
    │       └── ...
    └── varied_camera_2/
```

传了输出根目录时：

```text
<输出根目录>/
└── episode_00001/
    ├── _shared/
    │   └── query_frame_schedule_v3_<哈希>.json
    ├── varied_camera_1/
    │   ├── scene_meta.json
    │   └── samples/
    └── varied_camera_2/
```

### 6.2 Xperience 默认输出

Xperience 每个时间窗口生成一个输出目录，目录名包含 episode、相机和原始帧范围：

```text
<输出根目录>/
└── <uuid>__ep7__stereo_left__000000_000512/
    ├── _shared/
    │   └── query_frame_schedule_v3_<哈希>.json
    ├── lang.txt（如果原始 annotation 中有任务文字）
    ├── scene_meta.json
    └── samples/
        ├── <窗口名>_0.npz
        └── ...
```

Xperience 默认使用 `adapter_ref` 场景引用。`scene_meta.json` 会保存结构化源数据描述，包括原始数据集根目录、episode 路径、视频路径、相机名、窗口起止帧、原始图像分辨率和目标处理分辨率。

### 6.3 `scene_meta.json`

`scene_meta.json` 是每个相机输出目录的场景索引。核心内容包括：

| 内容 | 说明 |
| --- | --- |
| 输出协议版本 | 当前是 v2。 |
| 视频或相机输出名 | 用于定位当前输出对应的相机或窗口。 |
| 帧数、图像高宽 | 加载和处理后的场景尺寸。 |
| 原始高宽、处理高宽 | 若做过 resize，会记录原始分辨率和处理分辨率。 |
| 外参方向 | 当前保存为世界到相机。 |
| 每个加载帧的原始文件名 | 方便回查源帧。 |
| 每个加载帧对应的源帧编号 | 用于把输出时间轴映射回原始数据。 |
| 场景存储方式 | `source_ref`、`adapter_ref` 或 `cache`。 |
| 源 RGB、源深度、源几何路径 | `source_ref` 读取时用这些路径回到原始文件。 |
| 结构化源数据描述 | `adapter_ref` 读取 Xperience 时使用。 |
| 查询帧采样方式 | 记录是共享计划还是固定间隔回退。 |
| 查询帧计划路径和版本 | 对应 `_shared/query_frame_schedule_v3_<哈希>.json`。 |
| 每秒查询帧数量设置 | 记录最小值和最大值。 |
| 本地查询帧编号和源查询帧编号 | 同时记录加载后编号和原始源帧编号。 |
| 被丢弃的短尾查询帧 | 记录因为视频尾部太短而跳过的查询帧。 |

### 6.4 `samples/*.npz`

每个 NPZ 文件对应一个查询帧。字段名是落盘接口的一部分，下面按实际含义解释。

| 落盘字段名 | 含义 |
| --- | --- |
| `traj_uvz` | 主三维轨迹。每条轨迹在每个保存时间步上都有“横向像素、纵向像素、查询帧相机深度”。尾部补齐位置为无穷值。 |
| `keypoints` | 查询帧上的主二维网格点位置。 |
| `query_frame_index` | 当前样本从加载后视频的第几帧开始跟踪。 |
| `segment_frame_indices` | 这个样本真实覆盖的加载后帧编号，不包含尾部补齐时间步。 |
| `valid_steps` | 哪些保存时间步是真实帧。尾部补齐时间步为 false。 |
| `dense_query_count` | 完整二维网格的查询点数量。 |
| `tracked_query_count` | 实际送入 TAPIP3D 跟踪的主查询点数量；如果没有预筛选，通常等于完整网格数量。 |
| `support_grid_size` | 支撑网格边长。支撑轨迹不会保存，只保存这个配置方便复核。 |
| `traj_valid_mask` | 每条轨迹最终是否通过过滤。训练或可视化通常先看这个字段。 |
| `traj_supervision_mask` | 每条轨迹在每个时间步是否可作为监督使用。 |
| `traj_supervision_prefix_len` | 每条轨迹从开头连续可监督的时间步数量。 |
| `traj_supervision_count` | 每条轨迹总共有多少可监督时间步。 |
| `traj_depth_consistency_ratio` | 轨迹投影回原始深度图后，深度一致的比例。 |
| `traj_stable_depth_consistency_ratio` | 去掉高深度波动区域后，轨迹深度一致的比例。 |
| `traj_high_volatility_hit` | 轨迹是否经过高深度波动区域。 |
| `traj_volatility_exposure_ratio` | 轨迹时间步中落在高深度波动区域的比例。 |
| `traj_compare_frame_count` | 参与时域深度比较的时间步数量。 |
| `traj_stable_compare_frame_count` | 去掉高深度波动区域后参与比较的时间步数量。 |
| `traj_mask_reason_bits` | 轨迹被过滤的原因位标记。通过过滤的轨迹该值为 0。 |
| `traj_query_depth_rank` | 在部分机械臂相关过滤中，查询点深度由近到远的排序调试量。 |
| `traj_query_depth_edge_mask` | 查询点是否位于深度边缘。 |
| `traj_query_depth_patch_valid_ratio` | 查询点周围深度块里的有效深度比例。 |
| `traj_query_depth_patch_std` | 查询点周围深度块的深度标准差。 |
| `traj_query_depth_edge_risk_mask` | 查询点是否被判为深度边缘高风险。 |
| `traj_motion_extent` | 在监督时间步内，轨迹世界坐标最大位移幅度。 |
| `traj_motion_step_median` | 在监督时间步内，相邻时间步世界坐标位移的中位数。 |
| `traj_motion_extent_all_valid` | 在所有有效时间步内，轨迹世界坐标最大位移幅度。 |
| `traj_motion_step_median_all_valid` | 在所有有效时间步内，相邻时间步世界坐标位移的中位数。 |
| `traj_wrist_seed_mask` | 显式 wrist 相关过滤中的候选种子轨迹标记；默认 external 路径通常只是调试信息。 |
| `traj_manipulator_candidate_mask` | 显式机械臂相关过滤中的候选机械臂轨迹标记。 |
| `traj_manipulator_cluster_id` | 显式机械臂相关过滤中的二维空间连通区域编号。 |
| `traj_manipulator_component_size` | 显式机械臂相关过滤中的连通区域大小。 |
| `traj_manipulator_cluster_fallback_used` | 显式机械臂相关过滤是否触发了连通区域回退策略。 |
| `traj_pick_place_heatmap_hit_count` | 显式 pick-place 热力图路径中，轨迹命中热力图的次数。 |
| `traj_pick_place_heatmap_support_mask` | 显式 pick-place 热力图路径中，热力图支持的轨迹。 |
| `traj_pick_place_min_manipulator_distance` | 显式 pick-place 路径中，轨迹到机械臂参考轨迹的最小距离。 |
| `traj_pick_place_contact_mask` | 显式 pick-place 路径中，轨迹是否与机械臂接触或接近。 |
| `traj_pick_place_depth_guard_mask` | 显式 pick-place 路径中，轨迹是否通过深度保护条件。 |
| `traj_pick_place_delayed_contact_rescue_mask` | 无热力图 pick-place 路径中，被延迟接触逻辑救回的轨迹。 |
| `traj_pick_place_object_mask` | 显式 pick-place 路径中，被认为属于目标物体的轨迹。 |
| `original_frame_height` / `original_frame_width` | 原始图像分辨率。 |
| `processing_frame_height` / `processing_frame_width` | 实际送入模型处理的图像分辨率。 |
| `visibility` | 可选字段。开启保存可见性时，每条轨迹每个时间步的可见性判断。 |

默认训练或筛样最关键的字段通常是：

- `traj_uvz`
- `keypoints`
- `query_frame_index`
- `segment_frame_indices`
- `valid_steps`
- `traj_valid_mask`
- `traj_supervision_mask`

### 6.5 `cache` 模式额外输出

显式使用本地缓存时，每个相机输出目录会多出：

```text
scene.h5
scene_rgb.mp4
```

`scene.h5` 内容：

| H5 路径 | 含义 |
| --- | --- |
| `dense/depth` | 加载后的全视频深度图，单位为米。 |
| `camera/intrinsics` | 加载后的每帧相机内参。 |
| `camera/extrinsics_w2c` | 加载后的每帧世界到相机外参。 |

`scene_rgb.mp4` 是加载后的全视频 RGB 缓存。

### 6.6 批处理遥测输出

如果启用输出根目录或单独指定遥测目录，批处理会写：

```text
<遥测目录>/
├── _batch_run_summary.json
├── _camera_task_metrics.jsonl
├── _camera_task_profiles.jsonl
└── _hardware_telemetry.jsonl
```

含义：

| 文件 | 含义 |
| --- | --- |
| `_batch_run_summary.json` | 本次批处理的数据集、相机、任务数量、GPU、耗时和主要参数。 |
| `_camera_task_metrics.jsonl` | 每个相机任务或窗口任务的成功失败、耗时和输出目录。 |
| `_camera_task_profiles.jsonl` | 开启详细统计后，每个任务的加载、预处理、模型推理、保存阶段耗时。 |
| `_hardware_telemetry.jsonl` | 开启硬件采样后，周期记录 GPU、CPU 和 I/O 状态。 |

## 7. 旧版输出

`legacy` 输出仍可显式开启，但不是推荐默认协议。旧版目录大致为：

```text
<输出目录>/<视频名>/
├── images/
├── depth/
├── samples/
└── <视频名>.npz
```

主要差异：

- 查询帧 RGB 图像和深度图会被单独保存。
- 主轨迹字段名是 `traj`，而不是 `traj_uvz`。
- 会写一个全局 NPZ 供旧可视化工具使用。

新流程建议优先使用 `v2`，因为 `scene_meta.json` 能完整记录源数据引用、查询帧计划和坐标约定。

## 8. 可选语言指令生成

语言指令生成不是三维轨迹预处理的必需步骤。需要时可以单独运行：

```bash
cd text_generation
python generate_description.py \
  --episode_dir <episode 或数据集目录> \
  --images_subdir images \
  --provider openai \
  --model gpt-4o-mini \
  --skip_existing
```

输入：

- 一个 episode 的 `images/` 目录，或多个 episode 的父目录。
- 至少 5 张图像。

处理方式：

1. 按自然顺序列出图像。
2. 选择第一帧、最后一帧和中间均匀分布的三帧。
3. 把 5 张图发给视觉语言模型。
4. 解析模型返回的三个任务指令。

输出：

```text
<episode>/three_instructions.json
```

## 9. 校验和可视化

生成后可以先做结构检查：

```bash
python checker/batch_process_result_checker_3d.py <输出目录> --max-videos 1 --max-samples 3
python checker/batch_process_result_checker.py <输出目录> --max-videos 1 --max-samples 3
```

查看单个样本：

```bash
python scripts/visualization/visualize_single_image.py \
  --npz_path <相机输出目录>/samples/<样本文件>.npz \
  --port 8080
```

查看一个相机输出目录的动态三维点云和轨迹：

```bash
python scripts/visualization/visualize_3d_keypoint_animation.py \
  --episode_dir <相机输出目录> \
  --query_frame <查询帧编号> \
  --dense_pointcloud \
  --port 8080
```

如果只想确认输出是否完整，判断标准是：相机输出目录中存在 `scene_meta.json`，并且 `samples/` 下有至少一个查询帧 NPZ。

