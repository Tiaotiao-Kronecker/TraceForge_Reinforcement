# 可视化功能文档

## 概述

`visualize_single_image.py` 提供了交互式的3D轨迹可视化功能，支持实时调整渲染参数以优化显示效果和性能。

## 功能特性

### 1. 自适应 Keypoint 密度

可视化脚本自动适配不同的 `grid_size` 值：
- `grid_size=20`: 400 个 keypoints/trajectories
- `grid_size=30`: 900 个 keypoints/trajectories
- `grid_size=40`: 1600 个 keypoints/trajectories
- `grid_size=80`: 6400 个 keypoints/trajectories

### 2. 可调节的渲染参数

#### 轨迹控制
- **Number of trajectories** (滑块)
  - 范围: 1 到总轨迹数
  - 默认值: min(100, 总轨迹数)
  - 功能: 控制显示的轨迹数量，用于性能优化和视觉清晰度
  - 用途: 当轨迹数量很大时（如 grid_size=80），可以减少显示数量以提高性能

- **Track width** (滑块)
  - 范围: 0.5 - 10.0
  - 步长: 0.5
  - 默认值: 4.0
  - 功能: 调整轨迹线条的宽度

- **Track length** (滑块)
  - 范围: 1 到轨迹总长度
  - 默认值: min(30, 轨迹总长度)
  - 功能: 控制每条轨迹显示的长度（帧数）

#### Keypoint 控制
- **Number of keypoints** (滑块)
  - 范围: 1 到总 keypoint 数
  - 默认值: min(100, 总 keypoint 数)
  - 功能: 控制显示的 keypoint 数量
  - 用途: 独立于轨迹数量控制 keypoint 显示

- **Keypoint size** (滑块)
  - 范围: 0.001 - 0.1
  - 步长: 0.001
  - 默认值: 0.005
  - 功能: 调整 keypoint 点的大小

#### 其他控制
- **Point size** (滑块): 点云大小 (0.001 - 0.02)
- **Show point cloud** (复选框): 显示/隐藏点云
- **Show tracks** (复选框): 显示/隐藏轨迹
- **Show keypoints** (复选框): 显示/隐藏 keypoint
- **Show camera frustum** (复选框): 显示/隐藏相机视锥
- **Show world axes** (复选框): 显示/隐藏世界坐标轴

## 使用示例

### 基本使用

```bash
python visualize_single_image.py \
    --npz_path output_bridge_depth_grid80/00000/images0/samples/images0_0.npz \
    --image_path output_bridge_depth_grid80/00000/images0/images/images0_0.png \
    --depth_path output_bridge_depth_grid80/00000/images0/depth/images0_0.png \
    --port 8080
```

### 高密度 Keypoint 场景 (grid_size=80)

当使用 `grid_size=80` 时，会有 6400 个轨迹，建议：

1. **初始设置**: 将 "Number of trajectories" 设置为 100-500，以获得清晰的视图
2. **逐步增加**: 根据需要逐步增加显示数量
3. **性能优化**: 如果渲染变慢，减少显示数量或调整其他参数

### 交互式调整流程

1. 打开浏览器访问 `http://localhost:8080`
2. 在左侧面板找到 "Visualization" 文件夹
3. 调整以下参数：
   - 先设置 "Number of trajectories" 和 "Number of keypoints" 为较小值
   - 调整 "Track width" 和 "Keypoint size" 以获得清晰的视觉效果
   - 使用 "Show/Hide" 复选框控制不同元素的显示
4. 实时查看效果，根据需要继续调整

## 技术实现

### 动态更新机制

可视化脚本使用以下机制实现实时更新：

1. **update_trajectories()** 函数
   - 根据 "Number of trajectories" 滑块值动态创建/删除轨迹
   - 响应 "Track length" 和 "Track width" 的变化
   - 自动更新轨迹的可见性

2. **update_keypoints()** 函数
   - 根据 "Number of keypoints" 滑块值动态创建/删除 keypoint
   - 响应 "Keypoint size" 的变化
   - 自动更新 keypoint 的可见性

3. **回调函数**
   - 所有 GUI 控件都有对应的 `on_update` 回调
   - 滑块值改变时立即触发更新
   - 确保实时响应和流畅交互

### 性能优化

- **按需渲染**: 只渲染指定数量的轨迹和 keypoint
- **动态管理**: 使用 handle 管理，可以快速添加/删除元素
- **内存效率**: 不显示的元素会被移除，释放资源

## 更新历史

### 2024-02-06: 添加可调节的渲染参数

**新增功能**:
- 添加 "Number of trajectories" 滑块，控制显示的轨迹数量
- 添加 "Number of keypoints" 滑块，控制显示的 keypoint 数量
- 扩展 "Track width" 范围: 0.5-5.0 → 0.5-10.0
- 扩展 "Keypoint size" 范围: 0.005-0.05 → 0.001-0.1

**改进**:
- 重构渲染逻辑，使用函数式更新机制
- 实现实时响应，滑块变化立即更新可视化
- 优化性能，支持高密度 keypoint 场景（如 grid_size=80）

**使用场景**:
- 高密度 keypoint 可视化（grid_size=80, 6400 个轨迹）
- 性能优化（减少渲染元素数量）
- 视觉清晰度调整（独立控制轨迹和 keypoint 显示）

## 注意事项

1. **深度数据加载**: 可视化脚本优先从 `_raw.npz` 文件加载准确的深度值
2. **端口占用**: 如果默认端口 8080 被占用，可以改用其他端口（如 8081, 8082）
3. **浏览器兼容性**: 建议使用现代浏览器（Chrome, Firefox, Edge）
4. **性能考虑**: 当轨迹数量很大时，建议先减少显示数量，然后根据需要逐步增加

