# grid_size vs support_grid_size 覆盖范围对比

## 一、用户问题

**用户理解**：`grid_size`和`support_grid_size`都是针对整个屏幕区域？没有区别？

**答案**：**有区别**！虽然都覆盖图像区域，但覆盖范围不同：
- **查询点 (grid_size)**：**完全覆盖**整个图像，无边距
- **支持点 (support_grid_size)**：**有边距**，只覆盖图像中心区域（约95-97%）

---

## 二、代码实现对比

### 2.1 查询点生成 (grid_size)

**位置**：`infer.py` 第780-811行

**函数**：`create_uniform_grid_points`

```python
def create_uniform_grid_points(height, width, grid_size=20, device="cuda"):
    # Create uniform grid
    y_coords = np.linspace(0, height - 1, grid_size)  # 从0到height-1
    x_coords = np.linspace(0, width - 1, grid_size)   # 从0到width-1
    
    # Create meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Flatten and create points [N, 2]
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    return grid_tensor  # [1, grid_size*grid_size, 3]
```

**关键特点**：
- 使用 `np.linspace(0, height - 1, grid_size)`
- **覆盖范围**：从 `(0, 0)` 到 `(width-1, height-1)`
- **无边距**，完全覆盖整个图像

---

### 2.2 支持点生成 (support_grid_size)

**位置**：`utils/inference_utils.py` 第12-43行

**函数**：`get_grid_queries` → `get_points_on_a_grid`

```python
def get_grid_queries(grid_size: int, depths, intrinsics, extrinsics):
    image_size = depths.shape[-2:]
    xy = get_points_on_a_grid(grid_size, image_size)  # 调用get_points_on_a_grid
    # ... 后续处理（深度过滤、坐标转换等）
```

**底层函数**：`third_party/cotracker/model_utils.py` 第83-139行

```python
def get_points_on_a_grid(size, extent, center=None, device="cpu"):
    if center is None:
        center = [extent[0] / 2, extent[1] / 2]
    
    margin = extent[1] / 64  # 边距 = width / 64
    
    range_y = (margin - extent[0] / 2 + center[0], 
               extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], 
               extent[1] / 2 + center[1] - margin)
    
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
```

**关键特点**：
- 使用 `margin = extent[1] / 64`（即 `width / 64`）
- **覆盖范围**：从 `(margin, margin)` 到 `(width-margin, height-margin)`
- **有边距**，不覆盖边缘区域

---

## 三、覆盖范围对比

### 3.1 示例计算

**假设图像尺寸**：`518 × 322`（宽×高）

#### 查询点 (grid_size=20)

```python
y_coords = np.linspace(0, 321, 20)  # [0.0, 16.9, 33.8, ..., 321.0]
x_coords = np.linspace(0, 517, 20)  # [0.0, 27.2, 54.4, ..., 517.0]

覆盖范围:
  Y: [0.0, 321.0]  # 完全覆盖
  X: [0.0, 517.0]  # 完全覆盖
边距: 0 像素
覆盖比例: 100%
```

#### 支持点 (support_grid_size=16)

```python
margin = 518 / 64 = 8.1 像素

range_y = (8.1 - 322/2 + 161, 322/2 + 161 - 8.1)
        = (8.1, 313.9)
range_x = (8.1 - 518/2 + 259, 518/2 + 259 - 8.1)
        = (8.1, 509.9)

覆盖范围:
  Y: [8.1, 313.9]  # 有边距
  X: [8.1, 509.9]  # 有边距
边距: 8.1 像素
覆盖比例: Y=95.0%, X=96.9%
```

---

### 3.2 可视化对比

```
图像尺寸: 518 × 322

┌─────────────────────────────────────────────────────────┐
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │ ← 查询点覆盖
│  │                                                   │   │   整个区域
│  │  ┌───────────────────────────────────────────┐   │   │
│  │  │                                           │   │   │ ← 支持点覆盖
│  │  │        支持点覆盖区域                      │   │   │   中心区域
│  │  │        (有8.1像素边距)                    │   │   │   (95-97%)
│  │  │                                           │   │   │
│  │  └───────────────────────────────────────────┘   │   │
│  │                                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 四、为什么支持点要有边距？

### 4.1 深度估计的准确性

**边缘区域的问题**：
1. **深度不准确**：图像边缘的深度估计通常不准确
2. **相机畸变**：边缘区域受镜头畸变影响更大
3. **遮挡问题**：边缘更容易出现遮挡或无效深度

**解决方案**：
- 支持点只覆盖**中心区域**（有边距）
- 确保支持点的深度值**更准确**
- 提高跟踪质量

### 4.2 代码中的深度过滤

**位置**：`utils/inference_utils.py` 第24-30行

```python
d = depths[:, 0][torch.arange(depths.shape[0])[:, None], ji[..., 1], ji[..., 0]]
# 从深度图中提取支持点的深度值

mask = d[0] > 0  # 过滤无效深度（深度 > 0）
d = d[:, mask]
xy = xy[:, mask]  # 只保留有效深度的点
```

**关键点**：
- 支持点需要**有效的深度值**
- 边缘区域的深度可能无效（深度 = 0）
- 通过边距，减少无效深度的支持点

---

## 五、实际影响

### 5.1 点数对比

**查询点 (grid_size=20)**：
- 点数：`20 × 20 = 400`
- 覆盖：整个图像（100%）

**支持点 (support_grid_size=16)**：
- 初始点数：`16 × 16 = 256`
- 过滤后：可能少于256（取决于有效深度）
- 覆盖：图像中心区域（约95-97%）

### 5.2 使用场景

**查询点**：
- **用户指定的跟踪点**
- 需要覆盖整个图像（包括边缘）
- 保存到输出NPZ文件

**支持点**：
- **模型内部使用**
- 只覆盖中心区域（避免边缘深度不准确）
- 不保存到输出文件

---

## 六、代码流程

### 6.1 查询点生成流程

```
infer.py: process_single_video
  ↓
create_uniform_grid_points(height, width, grid_size=20)
  ↓
np.linspace(0, height-1, 20)  # 完全覆盖
np.linspace(0, width-1, 20)   # 完全覆盖
  ↓
返回: [1, 400, 3]  # 400个查询点
  ↓
保存到: samples/*.npz  # 用户可见
```

### 6.2 支持点生成流程

```
infer.py: process_single_video
  ↓
get_grid_queries(support_grid_size=16, depths, ...)
  ↓
get_points_on_a_grid(16, image_size)
  ↓
margin = width / 64  # 计算边距
linspace(margin, height-margin, 16)  # 有边距
linspace(margin, width-margin, 16)   # 有边距
  ↓
深度过滤: 只保留 depth > 0 的点
  ↓
返回: [N, 4]  # N ≤ 256个支持点（3D世界坐标）
  ↓
合并到查询点，输入模型
  ↓
模型内部使用，不保存
```

---

## 七、总结

### 7.1 关键区别

| 特性 | 查询点 (grid_size) | 支持点 (support_grid_size) |
|------|-------------------|---------------------------|
| **函数** | `create_uniform_grid_points` | `get_points_on_a_grid` |
| **覆盖范围** | 完全覆盖 (0, 0) 到 (W-1, H-1) | 中心区域，有边距 (margin, margin) 到 (W-margin, H-margin) |
| **边距** | 无边距 | 边距 = width / 64 |
| **覆盖比例** | 100% | 约95-97% |
| **点数** | grid_size² (如400) | support_grid_size² (如256)，可能更少（深度过滤后） |
| **用途** | 用户指定的跟踪点 | 模型内部辅助点 |
| **保存** | 保存到NPZ文件 | 不保存 |

### 7.2 为什么有区别？

1. **查询点**：用户需要跟踪整个图像的点，包括边缘
2. **支持点**：只用于模型内部，避免边缘深度不准确的问题

### 7.3 用户理解纠正

**❌ 错误理解**：都是针对整个屏幕区域，没有区别

**✅ 正确理解**：
- **查询点 (grid_size)**：完全覆盖整个图像，无边距
- **支持点 (support_grid_size)**：有边距，只覆盖图像中心区域（约95-97%）
- **区别**：覆盖范围不同，用途不同

---

## 八、实际示例

### 8.1 图像尺寸：518 × 322

**查询点 (grid_size=20)**：
```
Y范围: [0.0, 321.0]  # 完全覆盖
X范围: [0.0, 517.0]  # 完全覆盖
边距: 0 像素
点数: 400
```

**支持点 (support_grid_size=16)**：
```
边距: 8.1 像素 (518/64)
Y范围: [8.1, 313.9]  # 有边距
X范围: [8.1, 509.9]  # 有边距
覆盖比例: Y=95.0%, X=96.9%
点数: ≤256（深度过滤后可能更少）
```

### 8.2 图像尺寸：1920 × 1080

**查询点 (grid_size=20)**：
```
Y范围: [0.0, 1079.0]  # 完全覆盖
X范围: [0.0, 1919.0]  # 完全覆盖
边距: 0 像素
点数: 400
```

**支持点 (support_grid_size=16)**：
```
边距: 30.0 像素 (1920/64)
Y范围: [30.0, 1050.0]  # 有边距
X范围: [30.0, 1890.0]  # 有边距
覆盖比例: Y=94.4%, X=98.5%
点数: ≤256（深度过滤后可能更少）
```

---

## 九、代码位置总结

1. **查询点生成**：
   - `infer.py` 第780-811行：`create_uniform_grid_points`
   - 使用：第477-484行

2. **支持点生成**：
   - `utils/inference_utils.py` 第12-43行：`get_grid_queries`
   - 底层：`third_party/cotracker/model_utils.py` 第83-139行：`get_points_on_a_grid`
   - 使用：`infer.py` 第556行（通过`inference`函数）

---

## 十、关键结论

1. **覆盖范围不同**：
   - 查询点：完全覆盖整个图像
   - 支持点：有边距，只覆盖中心区域

2. **边距计算**：
   - 支持点边距 = `width / 64`
   - 查询点无边距

3. **用途不同**：
   - 查询点：用户指定的跟踪点，保存到输出
   - 支持点：模型内部辅助点，不保存

4. **设计原因**：
   - 支持点有边距，避免边缘深度不准确的问题
   - 查询点完全覆盖，满足用户跟踪整个图像的需求

