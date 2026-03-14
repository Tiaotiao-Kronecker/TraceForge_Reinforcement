# `support_grid_size` 作用机制详解

## 一、概述

`support_grid_size` 通过添加**额外的支持点（support points）**来辅助模型跟踪**查询点（query points）**。这些支持点不保存到最终结果中，但它们在推理过程中发挥关键作用。

---

## 二、完整工作流程

### 阶段1：生成支持点（`get_grid_queries`函数）

**位置**：`utils/inference_utils.py` 第12-43行

```python
def get_grid_queries(grid_size: int, depths, intrinsics, extrinsics):
    # 1. 在图像上创建均匀网格点
    image_size = depths.shape[-2:]
    xy = get_points_on_a_grid(grid_size, image_size)  # 16×16 = 256个点
    
    # 2. 从深度图中查询每个点的深度值
    ji = torch.round(xy).to(torch.int32)
    d = depths[:, 0][..., ji[..., 1], ji[..., 0]]  # 获取深度值
    
    # 3. 过滤无效点（深度为0的点）
    mask = d[0] > 0
    d = d[:, mask]
    xy = xy[:, mask]
    
    # 4. 将2D像素坐标转换为3D世界坐标
    #    使用相机内参和外参进行坐标变换
    inv_intrinsics0 = torch.linalg.inv(intrinsics[0, 0])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0, 0])
    
    # 像素坐标 → 归一化相机坐标 → 3D相机坐标 → 3D世界坐标
    xy_homo = torch.cat([xy, torch.ones_like(xy[:, :, :1])], dim=-1)
    xy_homo = torch.einsum('ij,bnj->bni', inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[:, :, :1])], dim=-1)
    world_coords = torch.einsum('ij,bnj->bni', inv_extrinsics0, local_coords_homo)
    
    # 5. 构建查询格式：[t, x, y, z] (t=0表示第一帧)
    queries = torch.cat([torch.zeros_like(xy[:, :, :1]), world_coords], dim=-1)
    return queries  # 形状: (1, N_support, 4)
```

**关键点**：
- 支持点从**第一帧的深度图**中生成
- 使用**真实的深度值**（不是估计值）
- 转换为**3D世界坐标**，与查询点格式一致
- 时间维度设为0（第一帧）

---

### 阶段2：合并查询点和支持点

**位置**：`utils/inference_utils.py` 第58-61行

```python
if grid_size != 0:
    # 生成支持点（16×16 = 256个）
    additional_queries = get_grid_queries(grid_size, ...)
    
    # 合并：查询点(400) + 支持点(256) = 656个点
    query_point = torch.cat([query_point, additional_queries], dim=1)
    # query_point形状: (1, 656, 4)
    # 前400个是查询点，后256个是支持点
    
    N_supports = additional_queries.shape[1]  # 256
```

**数据结构**：
```
query_point = [
    [查询点1: t, x, y, z],
    [查询点2: t, x, y, z],
    ...
    [查询点400: t, x, y, z],
    [支持点1: 0, x, y, z],  ← 从第一帧深度图生成
    [支持点2: 0, x, y, z],
    ...
    [支持点256: 0, x, y, z]
]
```

---

### 阶段3：模型推理（使用所有点）

**位置**：`utils/inference_utils.py` 第65-74行

```python
# 模型接收所有656个点（查询点+支持点）
preds, train_data_list = model(
    rgb_obs=video,           # RGB视频
    depth_obs=depths,        # 深度图
    query_point=query_point,  # 656个点（400查询 + 256支持）
    intrinsics=intrinsics,   # 相机内参
    extrinsics=extrinsics,   # 相机外参
    mode="inference",
    ...
)
```

**模型内部处理**（基于Transformer架构）：

1. **特征提取**：
   - 从RGB图像中提取特征图（feature maps）
   - 从深度图中提取深度信息

2. **相关性计算（Correlation）**：
   ```python
   # 在模型内部（BaseTrackerPredictor）
   fcorr_fn = CorrBlock(fmaps, num_levels=5, radius=4)
   fcorrs = fcorr_fn.corr_sample(track_feats, coords)
   ```
   - 计算每个点与周围区域的相关性
   - **支持点提供额外的空间上下文**，帮助模型理解场景结构

3. **Transformer处理**：
   ```python
   # 所有点（查询点+支持点）一起输入Transformer
   transformer_input = torch.cat([flows_emb, fcorrs_, track_feats_], dim=2)
   x = transformer(transformer_input)  # 处理所有656个点
   ```
   - Transformer可以**同时关注所有点**（自注意力机制）
   - 支持点帮助模型理解**空间关系**和**场景结构**

4. **迭代优化**：
   - 模型进行多次迭代（`num_iters=6`）
   - 每次迭代，支持点提供**稳定的参考**，帮助优化查询点的跟踪

---

### 阶段4：分离结果（只返回查询点）

**位置**：`utils/inference_utils.py` 第75-77行

```python
N_total = preds.coords.shape[2]  # 656（查询点+支持点）

# 只返回前400个查询点的结果，丢弃后256个支持点的结果
preds = preds.query_slice(slice(0, N_total - N_supports))
# 结果：只保留前400个点（查询点）
```

**关键操作**：
- `query_slice(slice(0, 400))`：只保留前400个点的轨迹
- 后256个支持点的轨迹被丢弃

---

## 三、支持点如何发挥作用？

### 3.1 提供空间上下文

**机制**：
- 支持点均匀分布在图像上（16×16网格）
- 覆盖整个场景，提供**全局空间信息**
- 帮助模型理解场景的**3D结构**

**示例**：
```
查询点：只关注特定区域（如物体上的点）
支持点：覆盖整个场景，提供背景和周围环境信息
```

### 3.2 辅助相关性计算

**机制**：
- 模型使用**相关性块（Correlation Block）**计算点之间的相似性
- 支持点提供**额外的参考点**，帮助匹配和跟踪

**代码位置**（`BaseTrackerPredictor`）：
```python
fcorr_fn = CorrBlock(fmaps, num_levels=5, radius=4)
fcorrs = fcorr_fn.corr_sample(track_feats, coords)
```

**作用**：
- 查询点可以通过与支持点的相关性来**验证自己的位置**
- 支持点提供**稳定的参考**，减少跟踪误差

### 3.3 约束优化过程

**机制**：
- 支持点使用**真实的深度值**（从深度图读取）
- 在迭代优化过程中，支持点提供**固定的约束**
- 帮助模型**校准**查询点的位置

**优势**：
- 支持点位置准确（基于真实深度）
- 提供**稳定的参考框架**
- 帮助模型理解**相机运动和场景变化**

### 3.4 Transformer自注意力机制

**机制**：
- Transformer可以**同时关注所有点**（查询点+支持点）
- 通过自注意力，查询点可以**学习支持点的特征**
- 支持点帮助模型理解**空间关系**

**代码位置**（`BaseTrackerPredictor`）：
```python
# 所有点一起输入Transformer
transformer_input = torch.cat([flows_emb, fcorrs_, track_feats_], dim=2)
x = transformer(transformer_input)  # 自注意力机制
```

**作用**：
- 查询点可以**关注**相关的支持点
- 支持点提供**上下文信息**，帮助查询点更好地跟踪

---

## 四、为什么支持点能提高跟踪质量？

### 4.1 更多的空间采样

**原理**：
- 查询点：400个（20×20网格）
- 支持点：256个（16×16网格）
- **总共656个点**，提供更密集的空间采样

**效果**：
- 更密集的点覆盖 → 更好的场景理解
- 更多的参考点 → 更稳定的跟踪

### 4.2 真实深度约束

**原理**：
- 支持点使用**真实的深度值**（从深度图读取）
- 查询点使用**估计的深度值**（可能不准确）

**效果**：
- 支持点提供**准确的3D位置参考**
- 帮助模型**校准**查询点的深度估计
- 减少深度估计误差

### 4.3 全局场景理解

**原理**：
- 支持点覆盖**整个场景**
- 查询点可能只关注**局部区域**

**效果**：
- 支持点提供**全局上下文**
- 帮助模型理解**场景的整体结构**
- 提高跟踪的**鲁棒性**

### 4.4 迭代优化的稳定性

**原理**：
- 模型进行多次迭代优化（`num_iters=6`）
- 支持点位置**固定**（基于真实深度）
- 查询点位置**变化**（逐步优化）

**效果**：
- 支持点提供**稳定的参考框架**
- 每次迭代，查询点可以**参考支持点**来优化
- 减少优化过程中的**漂移和误差**

---

## 五、具体示例

### 场景：跟踪一个移动的物体

**查询点（400个）**：
- 分布在物体表面
- 需要跟踪物体的运动

**支持点（256个）**：
- 分布在背景和周围环境
- 提供场景的3D结构信息

**模型处理**：
1. **特征提取**：从RGB和深度图中提取特征
2. **相关性计算**：
   - 查询点与支持点计算相关性
   - 支持点提供稳定的参考
3. **Transformer处理**：
   - 所有点一起输入Transformer
   - 查询点可以关注相关的支持点
   - 支持点提供空间上下文
4. **迭代优化**：
   - 每次迭代，查询点参考支持点优化位置
   - 支持点提供稳定的约束

**结果**：
- 查询点跟踪更准确
- 支持点不保存（只是辅助工具）

---

## 六、代码执行流程总结

```
1. 生成支持点（get_grid_queries）
   ├─ 创建16×16网格（256个点）
   ├─ 从第一帧深度图读取深度值
   └─ 转换为3D世界坐标

2. 合并点（_inference_with_grid）
   ├─ 查询点：400个（用户指定）
   ├─ 支持点：256个（自动生成）
   └─ 合并：656个点

3. 模型推理（model forward）
   ├─ 特征提取：RGB + 深度
   ├─ 相关性计算：所有点之间的相关性
   ├─ Transformer处理：自注意力机制
   └─ 迭代优化：6次迭代

4. 分离结果（query_slice）
   ├─ 总点数：656
   ├─ 查询点：前400个（保存）
   └─ 支持点：后256个（丢弃）
```

---

## 七、关键设计思想

### 7.1 为什么支持点不保存？

1. **只是辅助工具**：支持点的目的是帮助跟踪，不是最终结果
2. **节省存储**：保存656个点比400个点多64%的存储空间
3. **用户需求**：用户只关心查询点的轨迹

### 7.2 为什么支持点从第一帧生成？

1. **固定参考**：第一帧作为参考帧，支持点位置固定
2. **真实深度**：使用第一帧的真实深度，提供准确约束
3. **稳定性**：固定的支持点提供稳定的参考框架

### 7.3 为什么使用16×16网格？

1. **平衡**：256个点提供足够的空间覆盖，不会太多
2. **计算效率**：656个点（400+256）在计算上可接受
3. **经验值**：16是经过实验验证的有效值

---

## 八、总结

`support_grid_size` 通过以下机制发挥作用：

1. **生成支持点**：从第一帧深度图生成256个均匀分布的支持点
2. **合并输入**：将支持点与查询点合并，一起输入模型
3. **辅助跟踪**：
   - 提供空间上下文
   - 辅助相关性计算
   - 约束优化过程
   - 通过Transformer自注意力提供全局信息
4. **分离结果**：只返回查询点的轨迹，丢弃支持点

**核心思想**：支持点作为"辅助工具"，帮助模型更好地跟踪查询点，但不保存到最终结果中。

