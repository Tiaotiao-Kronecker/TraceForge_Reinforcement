#!/bin/bash
# 删除 4 个失败样本的已有输出，并重新推理（grid_size=80）

set -e
OUT_DIR="/home/wangchen/projects/TraceForge/output_bridge_depth_grid80"
DATASET="/usr/data/dataset/opt/dataset_temp/bridge_depth"
FAILED_IDS="03080 03082 05685 05837"
GRID_SIZE=80
FRAME_DROP_RATE=5
DEVICE="cuda:0"

# 项目根目录和 infer 脚本
PROJECT_ROOT="/home/wangchen/projects/TraceForge"
INFER_SCRIPT="$PROJECT_ROOT/scripts/batch_inference/infer.py"
PYTHON="${PYTHON:-/home/wangchen/.conda/envs/traceforge/bin/python}"

echo "=========================================="
echo "删除失败样本的已有输出并重新推理"
echo "=========================================="
echo "失败样本: $FAILED_IDS"
echo "grid_size: $GRID_SIZE"
echo "=========================================="

# 1. 删除已有输出
for tid in $FAILED_IDS; do
    d="$OUT_DIR/$tid"
    if [ -d "$d" ]; then
        echo "删除: $d"
        rm -rf "$d"
    fi
done
echo ""

# 2. 重新推理
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

for tid in $FAILED_IDS; do
    video_path="$DATASET/$tid/images0"
    depth_path="$DATASET/$tid/depth_images0"
    traj_out="$OUT_DIR/$tid"

    if [ ! -d "$video_path" ]; then
        echo "⚠️  跳过 $tid: 数据集无 images0 ($video_path)"
        continue
    fi
    if [ ! -d "$depth_path" ]; then
        echo "⚠️  跳过 $tid: 数据集无 depth_images0 ($depth_path)"
        continue
    fi

    echo "推理: $tid"
    $PYTHON "$INFER_SCRIPT" \
        --use_all_trajectories \
        --frame_drop_rate $FRAME_DROP_RATE \
        --grid_size $GRID_SIZE \
        --out_dir "$traj_out" \
        --video_path "$video_path" \
        --depth_path "$depth_path" \
        --device "$DEVICE"

    if [ -d "$traj_out/images0/images" ] && [ -n "$(ls -A "$traj_out/images0/images" 2>/dev/null)" ]; then
        echo "  ✅ $tid 完成"
    else
        echo "  ❌ $tid 仍失败"
    fi
    echo ""
done

echo "=========================================="
echo "完成"
echo "=========================================="
