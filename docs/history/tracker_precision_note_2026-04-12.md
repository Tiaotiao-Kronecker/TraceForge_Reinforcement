# 当前 3D tracker 精度路径说明

日期：2026-04-12

提交：`9e6bcbf700cf`（原分支 `feat/external-only-maintained-mode`，现已进入 `main`）

## 结论

当前 batch inference 主路径下，底层 3D tracker 默认按 `FP32` 运行，不是全局 `AMP/FP16` 或 `BF16`。

## 依据

### 1. 模型加载阶段没有降精度

- `scripts/batch_inference/infer.py`
  - 主入口使用 `load_model(args.checkpoint).to(args.device)`
  - 这里只做设备迁移，没有 `.half()`、`.bfloat16()` 或 `autocast`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`
  - batch 入口同样使用 `infer.load_model(args.checkpoint).to(args.device)`
- `utils/inference_utils.py`
  - `load_model()` 只负责 `models.from_pretrained(...)`、`set_eval_mode("raw")` 和 `model.eval()`
  - 没有任何全局 dtype cast

### 2. PointTracker3D 主实现显式偏向 float32

- `utils/common_utils.py`
  - `ensure_float32()` 会关闭 `torch.autocast`
  - 如果输入张量是 `float16` 或 `bfloat16`，会转回 `float32`，或者在不允许 cast 时直接报错
- `models/point_tracker_3d.py`
  - 多段核心逻辑使用 `with torch.autocast(device_type="cuda", enabled=False):`
  - 内部有显式断言：
  - `coords should be float32. we could use bfloat16 for delta, but not for coords`
  - `image_feats` 也会被转成 `torch.float32`

### 3. 当前结论针对的是主推理路径，不等于仓库所有实验代码都纯 FP32

- `models/SpaTrackV2/models/tracker3D/delta_utils/blocks.py`
  - 存在局部 flash-attn 分支，会临时把 `q/k/v` 转成 half
- 但这属于另一套实现内部的局部算子路径
- 不改变当前 TAPIP3D batch inference 主路径“默认按 FP32 跑”的结论

## 适用范围

这条结论针对当前分支上以下主入口：

- `scripts/batch_inference/infer.py`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`

如果后续为了吞吐专门引入统一的 `autocast`、`.half()`、`.bfloat16()` 或 checkpoint 级 dtype 改造，需要重新确认。
