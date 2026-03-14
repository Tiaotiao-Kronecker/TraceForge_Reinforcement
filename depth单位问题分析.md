# Depth单位问题分析 - 可视化看不到的原因

## 一、问题诊断

### 1.1 实际Depth数据

从`output4/images0/depth/images0_0_raw.npz`分析：
- **Depth范围**：205.2 - 806.2
- **Depth中位数**：479.5
- **单位判断**：**毫米（mm）**

### 1.2 代码期望

代码期望depth是**米（m）**单位：
- 估计的depth通常在0.5-1.0米范围
- `align_depth_scale`函数直接使用median计算scale

### 1.3 问题根源

**单位不匹配导致scale计算错误**：

```python
# utils/video_depth_pose_utils.py 第20行
scale = np.median(known_depth[valid_mask]) / np.median(pred_depth[valid_mask])
```

**实际情况**：
- `known_depth`（毫米）：median ≈ 480mm
- `pred_depth`（米）：median ≈ 0.75m
- **scale = 480 / 0.75 ≈ 640**

**结果**：
- 对齐后的depth = `pred_depth × 640` ≈ 480米（应该是0.48米）
- 轨迹坐标会异常大（放大640倍）
- 可视化时点云和轨迹都超出视野范围，看不到

---

## 二、Depth处理流程

### 2.1 Depth加载（infer.py 第435-439行）

```python
depth_tensor, _, _ = load_video_and_mask(
    depth_path, None, stride, args.max_num_frames, is_depth=True
)  # [T, H, W]
valid_depth = (depth_tensor > 0)
depth_tensor[~valid_depth] = 0
```

**问题**：`load_video_and_mask`函数（第705-764行）加载depth时：
- 如果是PNG文件，直接读取16位整数
- **没有单位转换**：毫米数据直接作为米使用

### 2.2 Depth Scale对齐（utils/video_depth_pose_utils.py 第93-105行）

```python
if known_depth is not None:
    # 1. 插值到估计depth的尺寸
    known_depth = torch.nn.functional.interpolate(...)
    
    # 2. Scale对齐（关键步骤）
    depth_npy, scale = align_video_depth_scale(depth_npy, known_depth)
    # scale = median(known_depth) / median(pred_depth)
    # 如果known_depth是毫米，scale会非常大（~500-800）
    
    # 3. 如果replace_with_known_depth=False，只做scale对齐，不替换
    if replace_with_known_depth:
        depth_npy = known_depth
        depth_conf = (known_depth > 0).astype(np.float32)
    
    # 4. 外参的平移部分也会被scale影响
    extrs_npy[:, :3, 3] *= scale  # 相机位置也被放大640倍！
```

**关键问题**：
1. **Scale计算错误**：如果known_depth是毫米，scale会异常大
2. **Depth被放大**：对齐后的depth = pred_depth × scale（异常大）
3. **相机位置被放大**：`extrs_npy[:, :3, 3] *= scale`（相机位置也被放大640倍）
4. **轨迹坐标异常**：基于错误的depth和相机参数，轨迹坐标会异常大

---

## 三、为什么可视化看不到？

### 3.1 点云问题

```python
# visualize_single_image.py 第189-191行
valid_mask = (points_xyz_ds[:, 2] > 0) & (points_xyz_ds[:, 2] < 10.0)
```

- 如果depth是480米（而不是0.48米），所有点都会被过滤掉（> 10米）
- 或者点云距离相机非常远，超出视野范围

### 3.2 轨迹坐标问题

```python
# visualize_single_image.py 第95-129行
# convert_image_coords_to_world函数
cam_point = np.array([x_norm * z, y_norm * z, z, 1.0])
world_point = c2w @ cam_point
```

- 如果z（深度）是480米而不是0.48米，轨迹坐标会异常大
- 相机位置也被放大640倍，导致整个场景尺度错误

### 3.3 验证

从NPZ文件分析：
```python
traj: shape=(400, 128, 3)
  traj min: 0.0
  traj max: 806.2  # 这个值异常大！应该是0.8米左右
  traj mean: 309.1  # 这个值也异常大！
```

**结论**：轨迹坐标确实异常大，证实了depth单位问题。

---

## 四、解决方案

### 方案1：在加载depth时转换为米（推荐）

修改`infer.py`第435-439行：

```python
# 加载depth
depth_tensor, _, _ = load_video_and_mask(
    depth_path, None, stride, args.max_num_frames, is_depth=True
)  # [T, H, W]

# 检查并转换单位：如果最大值>100，假设是毫米，转换为米
if depth_tensor.max() > 100:
    logger.info(f"Detected depth in millimeters (max: {depth_tensor.max():.1f}), converting to meters.")
    depth_tensor = depth_tensor / 1000.0

valid_depth = (depth_tensor > 0)
depth_tensor[~valid_depth] = 0
```

### 方案2：在scale对齐前转换

修改`utils/video_depth_pose_utils.py`第93-100行：

```python
if known_depth is not None:
    known_depth = torch.nn.functional.interpolate(...)
    known_depth = known_depth.cpu().numpy()
    
    # 检查并转换单位
    if known_depth.max() > 100:
        known_depth = known_depth / 1000.0  # 毫米转米
    
    depth_npy, scale = align_video_depth_scale(depth_npy, known_depth)
    ...
```

### 方案3：使用replace_with_known_depth=True（不推荐）

修改`infer.py`第450行：

```python
replace_with_known_depth=True,  # 直接使用known_depth，不做scale对齐
```

**缺点**：
- 会跳过scale对齐，可能影响轨迹质量
- 如果known_depth本身有问题，无法修正

---

## 五、验证修复

修复后，应该看到：
1. **Depth范围**：0.2 - 0.8米（而不是200-800毫米）
2. **Scale值**：约0.5-1.5（而不是500-800）
3. **轨迹坐标**：正常范围（0-1米左右）
4. **点云可见**：在10米范围内
5. **可视化正常**：点云和轨迹都能看到

---

## 六、总结

**根本原因**：
- Depth数据是**毫米单位**，但代码期望**米单位**
- Scale对齐计算时，scale值异常大（~640倍）
- 导致depth、相机位置、轨迹坐标都被异常放大
- 可视化时超出视野范围，看不到

**解决方案**：
- 在加载depth时检测单位，如果是毫米则转换为米
- 或者在使用known_depth前统一转换为米

**关键代码位置**：
1. `infer.py` 第435-439行：depth加载
2. `utils/video_depth_pose_utils.py` 第98-105行：scale对齐
3. `visualize_single_image.py` 第189-191行：点云过滤

