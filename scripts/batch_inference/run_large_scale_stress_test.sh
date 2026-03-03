#!/bin/bash
# 大规模压力测试脚本
# 用于重现累积性问题（如conda竞争）

# 配置
BASE_PATH="/usr/data/dataset/opt/dataset_temp/bridge_depth"
OUT_DIR="./stress_test_large_scale_$(date +%Y%m%d_%H%M%S)"
GPU_IDS="0,1,2,3,4,5"
MAX_WORKERS=6
MIN_TRAJS=2500  # 至少处理2500个traj以重现累积性问题
FRAME_DROP_RATE=5
GRID_SIZE=80

# 创建日志目录
LOG_DIR="./stress_test_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stress_test_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "大规模压力测试"
echo "=========================================="
echo "基础路径: $BASE_PATH"
echo "输出目录: $OUT_DIR"
echo "GPU IDs: $GPU_IDS"
echo "最大工作进程数: $MAX_WORKERS"
echo "最小traj数量: $MIN_TRAJS"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""

# 运行压力测试
python stress_test_batch_inference.py \
    --base_path "$BASE_PATH" \
    --out_dir "$OUT_DIR" \
    --gpu_id "$GPU_IDS" \
    --max_trajs "$MIN_TRAJS" \
    --min_trajs_for_issue "$MIN_TRAJS" \
    --max_workers "$MAX_WORKERS" \
    --frame_drop_rate "$FRAME_DROP_RATE" \
    --grid_size "$GRID_SIZE" \
    --monitor_interval 60 \
    --check_integrity \
    2>&1 | tee "$LOG_FILE"

# 检查退出码
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "测试完成，退出码: $EXIT_CODE"
echo "=========================================="

# 如果失败，显示最后50行日志
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "最后50行日志:"
    tail -50 "$LOG_FILE"
fi

exit $EXIT_CODE



