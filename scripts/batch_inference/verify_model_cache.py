#!/usr/bin/env python3
"""
验证HuggingFace模型缓存的完整性
"""
import os
import sys
from pathlib import Path
import json

def verify_model_cache(model_name="Yuxihenry/SpatialTrackerV2_Front"):
    """
    验证模型缓存是否完整
    
    Args:
        model_name: HuggingFace模型名称，如 "Yuxihenry/SpatialTrackerV2_Front"
    
    Returns:
        (is_complete, details_dict)
    """
    print("=" * 70)
    print(f"验证模型缓存完整性: {model_name}")
    print("=" * 70)
    
    # HuggingFace缓存目录
    hf_cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    
    # 模型缓存目录（HuggingFace会将 "/" 替换为 "--"）
    model_cache_dir = hf_cache_base / f"models--{model_name.replace('/', '--')}"
    
    details = {
        'model_name': model_name,
        'cache_dir': str(model_cache_dir),
        'exists': False,
        'has_snapshots': False,
        'has_refs': False,
        'snapshot_count': 0,
        'files': [],
        'total_size': 0,
        'can_load': False,
        'load_error': None
    }
    
    print(f"\n【1. 检查缓存目录】")
    print(f"缓存基础目录: {hf_cache_base}")
    print(f"模型缓存目录: {model_cache_dir}")
    
    if not hf_cache_base.exists():
        print(f"❌ HuggingFace缓存基础目录不存在: {hf_cache_base}")
        return False, details
    
    if not model_cache_dir.exists():
        print(f"❌ 模型缓存目录不存在: {model_cache_dir}")
        print(f"   说明: 模型尚未下载到缓存")
        return False, details
    
    details['exists'] = True
    print(f"✅ 模型缓存目录存在")
    
    # 检查子目录结构
    print(f"\n【2. 检查缓存结构】")
    
    # 检查snapshots目录（实际模型文件）
    snapshots_dir = model_cache_dir / "snapshots"
    if snapshots_dir.exists():
        details['has_snapshots'] = True
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        details['snapshot_count'] = len(snapshot_dirs)
        print(f"✅ snapshots目录存在，包含 {len(snapshot_dirs)} 个快照")
        
        if snapshot_dirs:
            latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
            print(f"   最新快照: {latest_snapshot.name}")
            
            # 列出快照中的文件
            snapshot_files = list(latest_snapshot.rglob("*"))
            file_count = len([f for f in snapshot_files if f.is_file()])
            dir_count = len([f for f in snapshot_files if f.is_dir()])
            print(f"   文件数: {file_count}, 目录数: {dir_count}")
            
            # 计算总大小
            total_size = sum(f.stat().st_size for f in snapshot_files if f.is_file())
            details['total_size'] = total_size
            details['files'] = [str(f.relative_to(latest_snapshot)) for f in snapshot_files if f.is_file()]
            
            print(f"   总大小: {total_size / (1024**3):.2f} GB")
            
            # 检查关键文件
            print(f"\n【3. 检查关键文件】")
            key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
            found_files = []
            for key_file in key_files:
                # 可能在根目录或子目录中
                found = list(latest_snapshot.rglob(key_file))
                if found:
                    found_files.append(key_file)
                    file_path = found[0]
                    file_size = file_path.stat().st_size
                    print(f"   ✅ {key_file}: {file_size / (1024**2):.2f} MB")
                else:
                    print(f"   ⚠️  {key_file}: 未找到")
            
            if len(found_files) == 0:
                print(f"   ❌ 未找到任何关键模型文件")
                return False, details
    else:
        print(f"❌ snapshots目录不存在")
        return False, details
    
    # 检查refs目录（版本引用）
    refs_dir = model_cache_dir / "refs"
    if refs_dir.exists():
        details['has_refs'] = True
        ref_files = list(refs_dir.glob("*"))
        print(f"✅ refs目录存在，包含 {len(ref_files)} 个引用")
        if ref_files:
            for ref_file in ref_files:
                try:
                    with open(ref_file, 'r') as f:
                        ref_content = f.read().strip()
                    print(f"   {ref_file.name} -> {ref_content}")
                except:
                    pass
    else:
        print(f"⚠️  refs目录不存在（可能不影响使用）")
    
    # 尝试加载模型验证
    print(f"\n【4. 尝试加载模型验证】")
    try:
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
        print(f"   正在尝试加载模型...")
        model = VGGT4Track.from_pretrained(model_name)
        print(f"   ✅ 模型加载成功！")
        details['can_load'] = True
        
        # 检查模型参数
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   模型参数数量: {param_count / 1e6:.2f}M")
        
        del model  # 释放内存
        
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        details['can_load'] = False
        details['load_error'] = str(e)
        return False, details
    
    print(f"\n" + "=" * 70)
    print(f"✅ 缓存验证通过！模型缓存完整且可用")
    print(f"=" * 70)
    
    return True, details

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="验证HuggingFace模型缓存完整性")
    parser.add_argument(
        "--model",
        type=str,
        default="Yuxihenry/SpatialTrackerV2_Front",
        help="模型名称（默认: Yuxihenry/SpatialTrackerV2_Front）"
    )
    
    args = parser.parse_args()
    
    is_complete, details = verify_model_cache(args.model)
    
    if is_complete:
        print(f"\n【总结】")
        print(f"✅ 模型缓存完整，可以直接使用")
        print(f"   缓存位置: {details['cache_dir']}")
        print(f"   缓存大小: {details['total_size'] / (1024**3):.2f} GB")
        print(f"   文件数量: {len(details['files'])}")
        print(f"\n建议: 可以直接运行批量推理，子进程会从缓存加载模型")
        return 0
    else:
        print(f"\n【总结】")
        print(f"❌ 模型缓存不完整或不存在")
        if details['load_error']:
            print(f"   加载错误: {details['load_error']}")
        print(f"\n建议:")
        print(f"1. 手动下载模型:")
        print(f"   conda run -n traceforge python -c \"")
        print(f"   from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track")
        print(f"   VGGT4Track.from_pretrained('{args.model}')\"")
        print(f"")
        print(f"2. 或让batch_infer.py自动预加载（会在开始前下载）")
        return 1

if __name__ == "__main__":
    sys.exit(main())

