#!/usr/bin/env python3
"""
分析批量推理失败的原因
"""
import re
import sys

def analyze_failures():
    print("=" * 70)
    print("批量推理失败原因分析")
    print("=" * 70)
    
    print("\n【错误类型】")
    print("SSLError: SSL连接失败，无法从HuggingFace下载模型")
    print("错误信息: 'SSL: UNEXPECTED_EOF_WHILE_READING' EOF occurred in violation of protocol")
    
    print("\n【失败统计】")
    print("成功: 1962/10501 (18.7%)")
    print("失败: 8539/10501 (81.3%)")
    print("失败率: 81.3%")
    
    print("\n【根本原因】")
    print("1. 每个子进程都在尝试从HuggingFace下载模型")
    print("   - VGGT4Wrapper.__init__() -> load_model() -> VGGT4Track.from_pretrained()")
    print("   - 模型路径: 'Yuxihenry/SpatialTrackerV2_Front'")
    print("")
    print("2. 大量并发SSL连接导致:")
    print("   - 网络连接不稳定")
    print("   - SSL握手失败")
    print("   - HuggingFace服务器可能限制并发连接")
    print("   - 连接被重置")
    
    print("\n【问题位置】")
    print("文件: utils/video_depth_pose_utils.py")
    print("行数: 69-70")
    print("代码:")
    print("  def load_model(self, checkpoint_path='Yuxihenry/SpatialTrackerV2_Front'):")
    print("      model = VGGT4Track.from_pretrained(checkpoint_path)")
    print("")
    print("每个infer.py子进程都会执行这段代码，导致:")
    print("  - 10501个任务 × 每个任务下载模型 = 大量并发下载")
    print("  - 即使HuggingFace有缓存，首次下载时仍会失败")
    
    print("\n【解决方案】")
    print("方案1: 预先下载模型（推荐）")
    print("  - 在主进程中预先下载模型到本地缓存")
    print("  - 子进程可以直接使用缓存，无需下载")
    print("")
    print("方案2: 添加重试机制")
    print("  - 在模型加载时添加重试逻辑")
    print("  - 指数退避重试")
    print("")
    print("方案3: 限制并发数")
    print("  - 减少同时运行的子进程数")
    print("  - 避免过多并发SSL连接")
    print("")
    print("方案4: 使用本地模型路径")
    print("  - 如果模型已下载，使用本地路径而不是HuggingFace路径")
    print("  - 避免网络请求")
    
    print("\n【建议的修复步骤】")
    print("1. 在主进程启动前，预先下载模型:")
    print("   python -c \"from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track; VGGT4Track.from_pretrained('Yuxihenry/SpatialTrackerV2_Front')\"")
    print("")
    print("2. 修改VGGT4Wrapper，添加重试机制")
    print("")
    print("3. 在batch_infer.py中添加模型预加载步骤")
    print("")
    print("4. 设置HuggingFace缓存环境变量")

if __name__ == "__main__":
    analyze_failures()

