# `visualize_3d_keypoint_animation.py` 3D 可视化数据需求分析

## 1. 文档目的

本文档面向后续接入 TraceForge 3D 可视化链路的同事，说明
`scripts/visualization/visualize_3d_keypoint_animation.py`
当前实际读取的数据格式、目录约定、字段要求和兼容逻辑。

目标不是复述脚本实现细节，而是回答两个问题：

1. 后续训练/推理产物至少要长成什么样，才能被这个脚本直接读起来。
2. 当前脚本对不同轨迹坐标格式做了哪些兼容，未来如果要稳定迭代，哪些字段应该显式提供。

补充实现逻辑可参考：

- `scripts/visualization/visualize_3d_keypoint_animation.py`
- `scripts/visualization/VISUALIZE_3D_ANIMATION_LOGIC.md`

## 2. 脚本的能力边界

该脚本负责做单个相机视角下的 3D keypoint 动画可视化，支持：

- 按查询帧加载 keypoint 轨迹并播放动画
- 显示稀疏 keypoint 点云
- 显示轨迹线
- 可选显示密集点云
- 可选将几何变换到首帧相机坐标系

该脚本不负责：

- 多相机联合显示
- 从原始视频直接推理轨迹
- 自动推断数据集语义
- 自动生成缺失的 RGB / depth / 相机参数

## 3. 脚本当前支持的三种读取模式

### 3.1 模式 A: 仅用 `sample NPZ` 播放 keypoint 动画

这是最基础的模式，也是后续动作-轨迹预测模型最容易对齐的最小数据契约。

触发条件：

- 传入 `--episode_dir`
- `episode_dir/samples/` 下存在 `video_name_*.npz`
- 不要求 `--dense_pointcloud`

此模式下，脚本至少需要：

- `samples/<video_name>_<query_frame>.npz`

此模式可以播放：

- keypoint 点动画
- 轨迹线动画

此模式不能稳定提供：

- 帧对齐的密集点云

### 3.2 模式 B: 首帧查询 + 主 `main NPZ` + 多帧密集点云

触发条件：

- 开启 `--dense_pointcloud`
- `episode_dir/<video_name>.npz` 存在
- `query_frame` 为空或 `0`

此模式下，脚本优先从主 NPZ 读取：

- `coords`
- `depths`
- `intrinsics`
- `extrinsics`
- 可选 `video`

此模式的特点：

- keypoint 轨迹与视频帧对齐
- 密集点云会随时间步更新
- 是当前脚本里最完整的 3D 场景动画模式

### 3.3 模式 C: 非首帧查询 + 主 `main NPZ` + 单帧密集点云

触发条件：

- 开启 `--dense_pointcloud`
- `episode_dir/<video_name>.npz` 存在
- `query_frame != 0`

此模式下：

- keypoint 轨迹仍然主要来自 `samples/<video_name>_<query_frame>.npz`
- 密集点云来自 `depth/<video_name>_<query_frame>_raw.npz`
- 相机参数来自 `main NPZ`

此模式的特点：

- 可以补充一帧静态密集点云
- 轨迹和密集点云不一定严格帧对齐
- 更像“某个查询帧轨迹 + 对应时刻背景点云”的展示

## 4. 目录与命名约定

脚本把 `--episode_dir` 当成“单个相机目录”。

关键约定：

- `video_name = basename(episode_dir)`
- 查询帧索引从 `samples/<video_name>_<frame_idx>.npz` 的文件名末尾解析
- 若没有显式传 `--query_frame`，默认取排序后的第一个 sample 文件

当前兼容的目录结构如下：

```text
<episode_dir>/                         # 例如 outputs_press_one_button_demo/varied_camera_1
  samples/
    <video_name>_0.npz
    <video_name>_15.npz
    <video_name>_30.npz
  depth/
    <video_name>_0_raw.npz
    <video_name>_15_raw.npz
  images/
    <video_name>_0.png
    <video_name>_15.png
  <video_name>.npz
```

示例：

```text
outputs_press_one_button_demo/varied_camera_1/
  samples/varied_camera_1_0.npz
  depth/varied_camera_1_0_raw.npz
  images/varied_camera_1_0.png
  varied_camera_1.npz
```

## 5. 最小必需输入

### 5.1 只做 keypoint 动画时

最小需要：

- `--episode_dir`
- `episode_dir/samples/<video_name>_*.npz`

其中每个 sample NPZ 至少应包含：

- `traj`
- `keypoints`
- `frame_index`

`valid_steps` 不是强制字段，缺失时脚本会根据 `traj` 中是否整步全为 `inf` 自动推断。

### 5.2 需要密集点云时

除 sample NPZ 外，至少还需要：

- `episode_dir/<video_name>.npz`

非首帧静态密集点云时，还需要：

- `episode_dir/depth/<video_name>_<frame_idx>_raw.npz`

RGB 图片不是硬依赖，但建议提供：

- `episode_dir/images/<video_name>_<frame_idx>.png`

其中：

- 首帧多帧密集点云主要依赖主 NPZ
- 非首帧静态密集点云主要依赖 `depth/*_raw.npz`

## 6. 数据文件格式要求

### 6.1 `sample NPZ` 格式

脚本直接读取的字段如下：

| 字段名 | 是否必需 | 形状 | 当前语义 |
| --- | --- | --- | --- |
| `traj` | 必需 | `(N, T, 3)` | 每个 keypoint 的 3 通道轨迹；当前 TraceForge 推理产物里通常是 `pixel_depth=(u,v,z)`，不是直接的世界坐标 |
| `keypoints` | 必需 | `(N, 2)` | 查询帧上的 2D 网格采样点 |
| `frame_index` | 必需 | `(1,)` | 当前 sample 对应的查询帧索引 |
| `valid_steps` | 可选 | `(T,)` | 每个时间步是否有效 |
| `traj_2d` | 非必需 | `(N, T, 2)` | 纯 2D 图像坐标轨迹；当前脚本不直接使用，但推理产物里通常会带上 |
| `image_path` | 非必需 | `(1,)` | 当前脚本不直接使用，但推理产物里通常会带上 |

说明：

- `N` 通常为网格采样点数，当前常见是 `80 x 80 = 6400`
- `T` 是轨迹时间长度，当前样例中常见为 `32`
- 当前仓库中的真实 sample 样例通常不包含 `valid_steps`
- `valid_steps` 缺失时，脚本会执行：
  `valid_steps = ~np.all(np.isinf(traj), axis=(0, 2))`

### 6.2 当前项目中的坐标系约定

这一节给出当前 TraceForge 推理产物的直接结论，避免后续同事混淆。

| 文件 | 字段 | 当前常见坐标语义 | 备注 |
| --- | --- | --- | --- |
| `sample NPZ` | `traj` | `pixel_depth = (u, v, z)` | 不是纯 2D，也不是直接的世界坐标 |
| `sample NPZ` | `traj_2d` | `image_uv = (u, v)` | 纯图像/屏幕坐标 |
| `main NPZ` | `coords` | `world_xyz = (x, y, z)` | 世界坐标，单位米 |

也就是说，当前项目里“轨迹”并不是单一格式：

- 主 `main NPZ` 里的 `coords` 是世界坐标系 3D 轨迹
- `samples/*.npz` 里的 `traj` 当前实现中通常是 `(u_pixel, v_pixel, z_depth)`
- `samples/*.npz` 里的 `traj_2d` 才是纯 2D 屏幕/图像坐标

这三者不能混用。

尤其需要强调：

- `sample NPZ.traj` 不是纯屏幕坐标，因为它包含第 3 个通道 `z`
- `sample NPZ.traj` 也不是直接的世界坐标，因为前两个通道通常仍处于像素坐标系
- 当前可视化脚本会在需要时把 `sample NPZ.traj` 从 `(u,v,z)` 反投影回世界坐标

### 6.3 `traj` 的兼容逻辑

这是当前接入时最关键的一点。

脚本当前兼容两类 `traj`：

1. 世界坐标 `world_xyz`
2. 像素加深度坐标 `pixel_depth = (u, v, z)`

当前脚本并没有显式读取“坐标系类型”字段，而是用一个启发式规则判断：

- 如果 `traj` 的有限值里，`x` 或 `y` 的绝对值最大值大于 200
- 且主 NPZ 存在

则认为 `traj` 很可能是 `(u_pixel, v_pixel, z_depth)`，
并用主 NPZ 当前帧的 `intrinsics + extrinsics(w2c)` 将它反投影为世界坐标。

这意味着：

- 当前脚本可以兼容已有推理产物
- 但对未来长期维护来说，这个判断规则不够稳健

建议未来显式提供一个元数据字段，例如：

- `traj_coordinate_mode = "world_xyz"` 或 `"pixel_depth"`

当前脚本尚未读取该字段，但需求文档应先把它纳入规范。

如果后续模型直接输出世界坐标轨迹，则可以让：

- `sample NPZ.traj` 直接保存 `world_xyz`

如果后续模型仍输出 `(u,v,z)`，则至少还需要保证可视化侧能够拿到：

- 当前查询帧对应的 `intrinsics`
- 当前查询帧对应的 `extrinsics(w2c)`

在当前 TraceForge 目录约定里，这些信息通常通过 `main NPZ` 提供。

### 6.4 `main NPZ` 格式

当使用密集点云、坐标转换或首帧相机归一化时，脚本会读取主 NPZ。

当前实际读取的字段如下：

| 字段名 | 是否必需 | 形状 | 当前语义 |
| --- | --- | --- | --- |
| `coords` | 首帧密集模式必需 | `(T_video, N, 3)` | 与视频帧对齐的 keypoint 轨迹，世界坐标 |
| `depths` | 首帧密集模式必需 | `(T_video, H, W)` | 深度图，单位米 |
| `intrinsics` | 必需 | `(T_cam, 3, 3)` | 相机内参 |
| `extrinsics` | 必需 | `(T_cam, 4, 4)` | 世界到相机，`w2c` |
| `video` | 可选 | `(T_video, H, W, 3)` | RGB 视频帧，范围通常为 `[0, 1]` |

当前脚本不依赖但主 NPZ 里常见的字段：

- `height`
- `width`
- `visibs`
- `unc_metric`

说明：

- 当前脚本默认 `extrinsics` 是 `w2c`
- 使用时会通过 `np.linalg.inv(extrinsics)` 得到 `c2w`
- 若 `coords/depths/intrinsics/extrinsics` 的时间长度不一致，脚本会截到最短长度

### 6.5 `depth/*_raw.npz` 格式

非首帧密集点云模式会读取：

- `depth/<video_name>_<frame_idx>_raw.npz`

格式要求很简单：

| 字段名 | 是否必需 | 形状 | 语义 |
| --- | --- | --- | --- |
| `depth` | 必需 | `(H, W)` | 深度图，单位米 |

### 6.6 `images/*.png` 格式

当前脚本只把 RGB 图像作为点云颜色来源。

要求：

- 文件名与 `video_name` 和 `frame_idx` 对齐
- 能被 `PIL.Image.open(...).convert("RGB")` 打开

如果主 NPZ 没有 `video` 字段，脚本会尝试从 `images/` 目录逐帧找图。

## 7. 当前真实样例

下面的样例来自当前仓库中的真实输出目录，用于说明“当前脚本实际读取到的数据长什么样”。

### 7.1 样例 A: `outputs_press_one_button_demo/varied_camera_1`

#### 目录示例

```text
outputs_press_one_button_demo/varied_camera_1/
  samples/varied_camera_1_0.npz
  samples/varied_camera_1_15.npz
  depth/varied_camera_1_0_raw.npz
  images/varied_camera_1_0.png
  varied_camera_1.npz
```

#### `samples/varied_camera_1_0.npz` 的真实字段

```text
image_path: shape=(1,), dtype=<U50
frame_index: shape=(1,), dtype=int64
keypoints: shape=(6400, 2), dtype=float32
traj: shape=(6400, 32, 3), dtype=float32
traj_2d: shape=(6400, 32, 2), dtype=float32
```

注意：该真实样例中没有 `valid_steps` 字段。

#### 当前读取到的实际内容特征

```text
frame_index = [0]
image_path = ['images/varied_camera_1_0.png']

keypoints[0:5] =
[[ 0.     0.   ]
 [16.19   0.   ]
 [32.38   0.   ]
 [48.57   0.   ]
 [64.759  0.   ]]

keypoints[-5:] =
[[1214.24  719.  ]
 [1230.43  719.  ]
 [1246.62  719.  ]
 [1262.81  719.  ]
 [1279.    719.  ]]

traj[0,0] = [0.0371 0.103  1.9988]
traj[0,1] = [0.1013 0.1713 1.9987]
```

解释：

- `keypoints` 明显是基于 `1280 x 720` 图像做的规则网格采样
- `N = 6400` 说明当前网格大小是 `80 x 80`
- `traj` 整体范围中 `x/y` 最大可接近 `1279/719`，说明它更像 `(u, v, z)` 而不是世界坐标
- 因此当前脚本会走“像素深度转世界坐标”的兼容分支

#### `varied_camera_1.npz` 的真实字段

```text
coords: shape=(32, 6400, 3), dtype=float32
extrinsics: shape=(50, 4, 4), dtype=float32
intrinsics: shape=(50, 3, 3), dtype=float32
height: shape=(), dtype=int64
width: shape=(), dtype=int64
depths: shape=(32, 720, 1280), dtype=float16
unc_metric: shape=(50, 720, 1280), dtype=float16
visibs: shape=(32, 6400, 1), dtype=bool
```

其中前几项的真实示例如下：

```text
height,width = 720, 1280
coords[0,0] = [-1.4681 -2.39    0.7715]

intrinsics[0] =
[[869.1169   0.     640.    ]
 [  0.     869.1169 360.    ]
 [  0.       0.       1.    ]]
```

解释：

- `coords` 的数值范围在米级，更像世界坐标
- `depths` 和 `coords` 的时间长度都是 `32`
- `intrinsics/extrinsics` 比 `coords` 更长，脚本会按最短长度截断使用

### 7.2 样例 B: `outputs_droid_hand_w2c/hand_camera`

#### `samples/hand_camera_0.npz` 的真实字段

```text
image_path: shape=(1,), dtype=<U50
frame_index: shape=(1,), dtype=int64
keypoints: shape=(6400, 2), dtype=float32
traj: shape=(6400, 32, 3), dtype=float32
traj_2d: shape=(6400, 32, 2), dtype=float32
```

注意：该真实样例中同样没有 `valid_steps` 字段。

示例内容：

```text
frame_index = [0]
image_path = ['images/hand_camera_0.png']

traj[0,0] = [-134.2906  652.3848    0.0005]
traj[0,1] = [661.6951 407.174    0.0001]
```

解释：

- `x/y` 已明显超出普通世界坐标范围
- `z` 很小，仍然符合“像素 + 深度”的特征
- 当前脚本会依赖主 NPZ 的 `intrinsics/extrinsics` 把这类轨迹转为世界坐标

#### `hand_camera.npz` 的真实字段

```text
coords: shape=(32, 6400, 3), dtype=float32
extrinsics: shape=(320, 4, 4), dtype=float32
intrinsics: shape=(320, 3, 3), dtype=float32
height: shape=(), dtype=int64
width: shape=(), dtype=int64
depths: shape=(32, 720, 1280), dtype=float16
unc_metric: shape=(320, 720, 1280), dtype=float16
visibs: shape=(32, 6400, 1), dtype=bool
```

说明：

- DROID 样例里 `intrinsics/extrinsics` 全视频长度为 `320`
- `coords/depths` 只有当前片段长度 `32`
- 脚本会自动以最短时间长度对齐

## 8. 对后续动作-轨迹预测模型输出的建议

### 8.1 最小可用版本

如果后续模型只需要支持 3D keypoint 动画，而不要求密集点云，建议保证每个 `episode_dir` 下至少能产出：

```text
<episode_dir>/
  samples/
    <video_name>_<query_frame>.npz
```

每个 sample NPZ 至少包含：

- `traj: (N, T, 3)`
- `keypoints: (N, 2)`
- `frame_index: (1,)`

如果无效时间步存在，推荐同时写出：

- `valid_steps: (T,)`

### 8.2 完整可视化版本

如果后续模型需要支持：

- 密集点云
- 坐标系归一化
- `(u,v,z)` 转世界坐标

则建议补齐：

```text
<episode_dir>/
  samples/
  depth/
  images/
  <video_name>.npz
```

主 NPZ 至少包含：

- `coords`
- `depths`
- `intrinsics`
- `extrinsics`

### 8.3 为后续脚本迭代建议新增的元数据

当前脚本能跑，不代表当前格式已经足够稳健。

为了让未来脚本减少启发式判断，建议在规范里增加以下字段，即便第一版脚本还没读取：

| 字段名 | 建议值 | 作用 |
| --- | --- | --- |
| `traj_coordinate_mode` | `"world_xyz"` / `"pixel_depth"` | 避免靠数值范围猜轨迹坐标系 |
| `traj_unit` | `"meter"` | 明确深度与坐标单位 |
| `extrinsics_mode` | `"w2c"` | 明确主 NPZ 的外参语义 |
| `image_height` | 整数 | 避免只靠 RGB 或主 NPZ 推断尺寸 |
| `image_width` | 整数 | 同上 |
| `video_name` | 字符串 | 避免目录名与文件名前缀强绑定 |
| `query_frame_index` | 整数 | 与文件名解析做双重校验 |

建议把这些字段视为“规范中的推荐字段”，哪怕当前版本脚本还未消费。

## 9. 推荐的数据契约

如果现在要为“动作-轨迹预测模型 -> 3D 可视化脚本”定义一份接口规范，建议分成两层。

### 9.1 强制字段

- `samples/<video_name>_<query_frame>.npz`
- `traj: (N, T, 3)`
- `keypoints: (N, 2)`
- `frame_index: (1,)`

### 9.2 推荐字段

- `valid_steps: (T,)`
- `traj_coordinate_mode`
- `traj_unit`
- 主 `main NPZ`
- `depth/*_raw.npz`
- `images/*.png`

## 10. 结论

从 `visualize_3d_keypoint_animation.py` 的当前实现来看，真正的最小输入并不复杂：

- 一个 `episode_dir`
- 一组按查询帧命名的 `sample NPZ`

但如果要把它作为后续统一的 3D 可视化入口，仅靠最小输入还不够。
为了让脚本后续可以持续扩展而不依赖启发式逻辑，建议从现在开始把以下三件事纳入规范：

1. 显式声明 `traj` 的坐标语义
2. 显式声明 `extrinsics` 的语义为 `w2c` 还是 `c2w`
3. 在需要密集点云时，稳定提供 `main NPZ + depth_raw + RGB`

这样后续无论是训练输出、推理输出，还是动作-轨迹预测模型的中间结果，都能更稳定地接入同一套 3D 可视化脚本。
