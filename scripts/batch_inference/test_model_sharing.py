#!/usr/bin/env python3
"""
测试模型缓存共享机制
"""
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch

def load_model_in_subprocess(process_id):
    """在子进程中加载模型"""
    print(f"[进程 {process_id}] 开始加载模型...")
    start_time = time.time()
    
    try:
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
        model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        elapsed = time.time() - start_time
        print(f"[进程 {process_id}] ✅ 模型加载成功 (耗时: {elapsed:.2f}秒)")
        
        # 检查是否从缓存加载
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache = hf_cache / "models--Yuxihenry--SpatialTrackerV2_Front"
        if model_cache.exists():
            print(f"[进程 {process_id}] ✅ 使用缓存: {model_cache}")
        else:
            print(f"[进程 {process_id}] ⚠️  未找到缓存（可能刚下载）")
        
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[进程 {process_id}] ❌ 模型加载失败 (耗时: {elapsed:.2f}秒): {e}")
        return False, elapsed

if __name__ == "__main__":
    print("=" * 70)
    print("测试模型缓存共享机制")
    print("=" * 70)
    
    print("\n【测试场景】")
    print("模拟多个子进程同时加载模型，验证:")
    print("1. 如果模型已缓存，子进程是否可以直接使用")
    print("2. 加载时间是否很快（说明使用了缓存）")
    print("3. 是否会有并发下载问题")
    
    # 检查模型是否已缓存
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = hf_cache / "models--Yuxihenry--SpatialTrackerV2_Front"
    
    print(f"\n【缓存状态】")
    if model_cache.exists():
        print(f"✅ 模型已缓存: {model_cache}")
        print(f"   预期: 子进程应该快速加载（<5秒）")
    else:
        print(f"❌ 模型未缓存")
        print(f"   预期: 第一个子进程会下载，其他子进程可能失败")
    
    print(f"\n【开始测试】")
    print(f"启动3个子进程同时加载模型...")
    
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(load_model_in_subprocess, i) for i in range(3)]
        results = [f.result() for f in futures]
    
    print(f"\n【测试结果】")
    success_count = sum(1 for success, _ in results if success)
    avg_time = sum(elapsed for _, elapsed in results) / len(results)
    
    print(f"成功: {success_count}/{len(results)}")
    print(f"平均加载时间: {avg_time:.2f}秒")
    
    if model_cache.exists() and avg_time < 5:
        print(f"✅ 结论: 模型从缓存加载，子进程可以共享缓存")
    elif model_cache.exists() and avg_time > 10:
        print(f"⚠️  结论: 虽然模型已缓存，但加载时间较长")
    else:
        print(f"❌ 结论: 模型未缓存或加载失败")

