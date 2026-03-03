#!/bin/bash
# 检查批量推理结果的脚本
# 用于在推理完成后检查输出完整性、错误统计等

if [ $# -lt 1 ]; then
    echo "用法: $0 <输出目录> [日志文件]"
    echo "示例: $0 ./output_bridge_depth_full_20260206_120000 ./batch_inference_logs/full_batch_20260206_120000.log"
    exit 1
fi

OUT_DIR="$1"
LOG_FILE="${2:-}"

echo "=========================================="
echo "批量推理结果检查"
echo "=========================================="
echo "输出目录: $OUT_DIR"
if [ -n "$LOG_FILE" ]; then
    echo "日志文件: $LOG_FILE"
fi
echo "=========================================="
echo ""

# 检查输出目录是否存在
if [ ! -d "$OUT_DIR" ]; then
    echo "❌ 输出目录不存在: $OUT_DIR"
    exit 1
fi

# 1. 统计输出目录数量
echo "1. 输出目录统计"
echo "----------------------------------------"
OUTPUT_COUNT=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "输出目录中的traj数量: $OUTPUT_COUNT"

# 检查空目录
EMPTY_DIRS=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -empty)
EMPTY_COUNT=$(echo "$EMPTY_DIRS" | grep -v '^$' | wc -l)
if [ $EMPTY_COUNT -gt 0 ]; then
    echo "⚠️  发现 $EMPTY_COUNT 个空输出目录:"
    echo "$EMPTY_DIRS" | head -10
    if [ $EMPTY_COUNT -gt 10 ]; then
        echo "... (还有 $((EMPTY_COUNT - 10)) 个)"
    fi
else
    echo "✅ 所有输出目录都有内容"
fi
echo ""

# 2. 检查文件完整性（抽样检查）
echo "2. 文件完整性检查（随机抽样10个traj）"
echo "----------------------------------------"
SAMPLE_DIRS=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | shuf | head -10)
CORRUPTED=0
MISSING_FILES=0

for dir in $SAMPLE_DIRS; do
    traj_name=$(basename "$dir")
    
    # 检查必要的子目录
    images_dir="$dir/images0/images"
    depth_dir="$dir/images0/depth"
    samples_dir="$dir/images0/samples"
    
    issues=()
    
    if [ ! -d "$images_dir" ] || [ -z "$(ls -A "$images_dir" 2>/dev/null)" ]; then
        issues+=("缺少images")
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
    
    if [ ! -d "$depth_dir" ] || [ -z "$(ls -A "$depth_dir" 2>/dev/null)" ]; then
        issues+=("缺少depth")
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
    
    if [ ! -d "$samples_dir" ] || [ -z "$(ls -A "$samples_dir" 2>/dev/null)" ]; then
        issues+=("缺少samples")
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
    
    # 检查文件是否可读
    if [ -d "$images_dir" ]; then
        sample_img=$(find "$images_dir" -name "*.png" | head -1)
        if [ -n "$sample_img" ]; then
            if ! file "$sample_img" | grep -q "PNG\|image"; then
                issues+=("图像文件损坏")
                CORRUPTED=$((CORRUPTED + 1))
            fi
        fi
    fi
    
    if [ ${#issues[@]} -gt 0 ]; then
        echo "⚠️  $traj_name: ${issues[*]}"
    fi
done

if [ $CORRUPTED -eq 0 ] && [ $MISSING_FILES -eq 0 ]; then
    echo "✅ 抽样检查通过，所有文件完整"
fi
echo ""

# 3. 如果提供了日志文件，分析错误
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo "3. 日志文件分析"
    echo "----------------------------------------"
    
    # 统计成功/失败
    SUCCESS_COUNT=$(grep -c "✅.*处理完成" "$LOG_FILE" 2>/dev/null || echo "0")
    FAIL_COUNT=$(grep -c "❌.*处理失败" "$LOG_FILE" 2>/dev/null || echo "0")
    TOTAL_COUNT=$((SUCCESS_COUNT + FAIL_COUNT))
    
    echo "总traj数: $TOTAL_COUNT"
    echo "成功: $SUCCESS_COUNT"
    echo "失败: $FAIL_COUNT"
    
    if [ $TOTAL_COUNT -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_COUNT" | bc)
        echo "成功率: ${SUCCESS_RATE}%"
    fi
    echo ""
    
    # 检查conda错误
    CONDA_ERRORS=$(grep -i "conda\|An unexpected error has occurred" "$LOG_FILE" 2>/dev/null | wc -l)
    if [ $CONDA_ERRORS -gt 0 ]; then
        echo "⚠️  发现 $CONDA_ERRORS 个conda相关错误"
        echo "前5个conda错误:"
        grep -i "conda\|An unexpected error has occurred" "$LOG_FILE" 2>/dev/null | head -5 | sed 's/^/  /'
    else
        echo "✅ 未发现conda相关错误"
    fi
    echo ""
    
    # 检查其他错误类型
    CUDA_ERRORS=$(grep -i "cuda.*error\|illegal memory\|out of memory" "$LOG_FILE" 2>/dev/null | wc -l)
    SSL_ERRORS=$(grep -i "ssl.*error\|SSLError\|UNEXPECTED_EOF" "$LOG_FILE" 2>/dev/null | wc -l)
    
    if [ $CUDA_ERRORS -gt 0 ] || [ $SSL_ERRORS -gt 0 ]; then
        echo "其他错误统计:"
        [ $CUDA_ERRORS -gt 0 ] && echo "  CUDA错误: $CUDA_ERRORS"
        [ $SSL_ERRORS -gt 0 ] && echo "  SSL错误: $SSL_ERRORS"
        echo ""
    fi
    
    # 按时间段分析错误率
    echo "4. 累积性错误分析（按traj数量分段）"
    echo "----------------------------------------"
    python3 << EOF
import re
from collections import defaultdict

try:
    with open("$LOG_FILE", 'r') as f:
        lines = f.readlines()
    
    segments = {
        '0-1000': {'success': 0, 'fail': 0, 'conda_errors': 0},
        '1000-2000': {'success': 0, 'fail': 0, 'conda_errors': 0},
        '2000-5000': {'success': 0, 'fail': 0, 'conda_errors': 0},
        '5000+': {'success': 0, 'fail': 0, 'conda_errors': 0}
    }
    
    traj_count = 0
    for line in lines:
        if '✅' in line and '处理完成' in line:
            traj_count += 1
            if traj_count <= 1000:
                segments['0-1000']['success'] += 1
            elif traj_count <= 2000:
                segments['1000-2000']['success'] += 1
            elif traj_count <= 5000:
                segments['2000-5000']['success'] += 1
            else:
                segments['5000+']['success'] += 1
        elif '❌' in line and '处理失败' in line:
            traj_count += 1
            if traj_count <= 1000:
                segments['0-1000']['fail'] += 1
            elif traj_count <= 2000:
                segments['1000-2000']['fail'] += 1
            elif traj_count <= 5000:
                segments['2000-5000']['fail'] += 1
            else:
                segments['5000+']['fail'] += 1
        
        if 'conda' in line.lower() or 'An unexpected error has occurred' in line:
            if traj_count <= 1000:
                segments['0-1000']['conda_errors'] += 1
            elif traj_count <= 2000:
                segments['1000-2000']['conda_errors'] += 1
            elif traj_count <= 5000:
                segments['2000-5000']['conda_errors'] += 1
            else:
                segments['5000+']['conda_errors'] += 1
    
    for seg_name, stats in segments.items():
        total = stats['success'] + stats['fail']
        if total > 0:
            fail_rate = stats['fail'] / total * 100
            conda_rate = stats['conda_errors'] / total * 100
            print(f"{seg_name}: 总数={total}, 成功={stats['success']}, 失败={stats['fail']}, "
                  f"失败率={fail_rate:.1f}%, conda错误={stats['conda_errors']} ({conda_rate:.1f}%)")
        else:
            print(f"{seg_name}: 暂无数据")
except Exception as e:
    print(f"分析日志时出错: {e}")
EOF
    echo ""
fi

# 5. 总结
echo "=========================================="
echo "检查完成"
echo "=========================================="
echo "输出目录: $OUT_DIR"
echo "输出traj数量: $OUTPUT_COUNT"
if [ $EMPTY_COUNT -gt 0 ]; then
    echo "⚠️  空目录数量: $EMPTY_COUNT"
fi
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    if [ $CONDA_ERRORS -eq 0 ]; then
        echo "✅ 未发现conda错误（修复可能有效）"
    else
        echo "⚠️  发现conda错误，需要进一步调查"
    fi
fi
echo ""



