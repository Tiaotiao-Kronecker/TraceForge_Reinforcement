#!/usr/bin/env python3
"""
批量推理压力测试脚本

用于测试批量推理脚本在高并发情况下的稳定性和性能。

用法:
    python stress_test_batch_inference.py \
        --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
        --out_dir ./stress_test_output \
        --gpu_id 0,1,2,3,4,5 \
        --max_trajs 100 \
        --max_workers 6 \
        --check_integrity
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from collections import defaultdict
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return None

def check_output_integrity(traj_out_dir, traj_name):
    """
    检查输出文件的完整性
    
    返回:
        {
            'exists': bool,
            'has_images': bool,
            'has_depth': bool,
            'has_samples': bool,
            'file_count': int,
            'errors': list
        }
    """
    result = {
        'exists': False,
        'has_images': False,
        'has_depth': False,
        'has_samples': False,
        'file_count': 0,
        'errors': []
    }
    
    try:
        if not os.path.exists(traj_out_dir):
            result['errors'].append(f"输出目录不存在: {traj_out_dir}")
            return result
        
        result['exists'] = True
        
        # 检查子目录结构
        images_dir = os.path.join(traj_out_dir, "images0", "images")
        depth_dir = os.path.join(traj_out_dir, "images0", "depth")
        samples_dir = os.path.join(traj_out_dir, "images0", "samples")
        
        # 检查images目录
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            if image_files:
                result['has_images'] = True
                result['file_count'] += len(image_files)
            else:
                result['errors'].append(f"images目录为空: {images_dir}")
        else:
            result['errors'].append(f"images目录不存在: {images_dir}")
        
        # 检查depth目录
        if os.path.exists(depth_dir):
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png') or f.endswith('.npz')]
            if depth_files:
                result['has_depth'] = True
                result['file_count'] += len(depth_files)
            else:
                result['errors'].append(f"depth目录为空: {depth_dir}")
        else:
            result['errors'].append(f"depth目录不存在: {depth_dir}")
        
        # 检查samples目录
        if os.path.exists(samples_dir):
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.npz')]
            if sample_files:
                result['has_samples'] = True
                result['file_count'] += len(sample_files)
            else:
                result['errors'].append(f"samples目录为空: {samples_dir}")
        else:
            result['errors'].append(f"samples目录不存在: {samples_dir}")
        
        # 检查文件是否损坏（随机抽样检查）
        if result['has_images']:
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            if image_files:
                sample_file = os.path.join(images_dir, image_files[0])
                try:
                    from PIL import Image
                    img = Image.open(sample_file)
                    img.verify()
                except Exception as e:
                    result['errors'].append(f"图像文件损坏: {sample_file} - {e}")
        
        if result['has_depth']:
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
            if depth_files:
                sample_file = os.path.join(depth_dir, depth_files[0])
                try:
                    from PIL import Image
                    img = Image.open(sample_file)
                    img.verify()
                except Exception as e:
                    result['errors'].append(f"深度文件损坏: {sample_file} - {e}")
        
        if result['has_samples']:
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.npz')]
            if sample_files:
                sample_file = os.path.join(samples_dir, sample_files[0])
                try:
                    import numpy as np
                    data = np.load(sample_file)
                    # 检查必要的键
                    required_keys = ['traj', 'traj_2d', 'keypoints']
                    for key in required_keys:
                        if key not in data:
                            result['errors'].append(f"NPZ文件缺少键: {sample_file} - {key}")
                except Exception as e:
                    result['errors'].append(f"NPZ文件损坏: {sample_file} - {e}")
    
    except Exception as e:
        result['errors'].append(f"检查完整性时出错: {e}")
    
    return result

def check_duplicate_files(output_dir):
    """检查是否有重复文件（通过文件哈希）"""
    file_hashes = defaultdict(list)
    duplicate_count = 0
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(('.png', '.npz')):
                file_path = os.path.join(root, file)
                file_hash = calculate_file_hash(file_path)
                if file_hash:
                    file_hashes[file_hash].append(file_path)
    
    # 找出重复文件
    duplicates = {hash_val: paths for hash_val, paths in file_hashes.items() if len(paths) > 1}
    duplicate_count = sum(len(paths) - 1 for paths in duplicates.values())
    
    return duplicate_count, duplicates

def run_stress_test(args):
    """运行压力测试"""
    
    print("=" * 80)
    print("批量推理压力测试")
    print("=" * 80)
    print(f"基础路径: {args.base_path}")
    print(f"输出目录: {args.out_dir}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"最大traj数量: {args.max_trajs}")
    print(f"最大工作进程数: {args.max_workers}")
    print(f"检查完整性: {args.check_integrity}")
    print(f"监控间隔: {args.monitor_interval}秒")
    print(f"最小测试traj数: {args.min_trajs_for_issue}")
    if args.min_trajs_for_issue:
        print(f"⚠️  将运行至少 {args.min_trajs_for_issue} 个traj以重现累积性问题")
    print("=" * 80)
    
    # 构建batch_infer.py命令
    batch_infer_script = os.path.join(os.path.dirname(__file__), "batch_infer.py")
    
    cmd = [
        sys.executable, batch_infer_script,
        "--base_path", args.base_path,
        "--out_dir", args.out_dir,
        "--use_all_trajectories",
        "--frame_drop_rate", str(args.frame_drop_rate),
        "--skip_existing",
    ]
    
    if args.gpu_id:
        cmd.extend(["--gpu_id", args.gpu_id])
    
    if args.max_trajs:
        cmd.extend(["--max_trajs", str(args.max_trajs)])
    
    if args.max_workers:
        cmd.extend(["--max_workers", str(args.max_workers)])
    
    if args.grid_size:
        cmd.extend(["--grid_size", str(args.grid_size)])
    
    # 记录开始时间
    start_time = time.time()
    last_monitor_time = start_time
    
    # 运行批量推理
    print("\n开始运行批量推理...")
    print(f"命令: {' '.join(cmd)}")
    if args.min_trajs_for_issue:
        print(f"⚠️  目标：至少处理 {args.min_trajs_for_issue} 个traj以重现累积性问题")
    print("-" * 80)
    
    # 如果启用监控，使用实时输出模式
    if args.monitor_interval > 0 and args.min_trajs_for_issue:
        print(f"📊 每 {args.monitor_interval} 秒监控一次运行状态...")
        # 使用实时输出模式，以便监控进度
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲
        )
        
        # 实时读取输出并监控
        stdout_lines = []
        stderr_lines = []
        monitor_thread_running = True
        
        def monitor_progress():
            """监控线程：定期检查进度"""
            nonlocal last_monitor_time
            while monitor_thread_running:
                time.sleep(args.monitor_interval)
                current_time = time.time()
                elapsed = current_time - start_time
                
                # 检查输出目录中的traj数量
                if os.path.exists(args.out_dir):
                    traj_dirs = [d for d in Path(args.out_dir).iterdir() if d.is_dir()]
                    completed_count = len(traj_dirs)
                    
                    print(f"\n📊 [监控] 运行时间: {elapsed/60:.1f}分钟 | 已完成: {completed_count}/{args.max_trajs or '?'} traj")
                    
                    # 检查是否有失败迹象
                    if completed_count > 0:
                        # 检查最近的输出
                        recent_dirs = sorted(traj_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
                        print(f"   最近完成的traj: {', '.join([d.name for d in recent_dirs[:3]])}")
        
        # 启动监控线程
        import threading
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # 读取输出
        def read_output(pipe, lines_list):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        lines_list.append(line)
                        # 实时打印关键信息
                        if any(keyword in line.lower() for keyword in ['成功', '失败', '处理完成', '处理失败', 'error', 'exception']):
                            print(line.rstrip())
            except Exception as e:
                print(f"读取输出时出错: {e}")
        
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines), daemon=True)
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_lines), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # 等待进程完成
        try:
            return_code = process.wait()
            monitor_thread_running = False
            
            # 等待输出线程完成
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            result = type('Result', (), {
                'returncode': return_code,
                'stdout': ''.join(stdout_lines),
                'stderr': ''.join(stderr_lines),
            })()
        except KeyboardInterrupt:
            print("\n⚠️  用户中断，正在终止进程...")
            process.terminate()
            process.wait()
            monitor_thread_running = False
            raise
    else:
        # 标准模式：捕获所有输出
        try:
            result = subprocess.run(
                cmd,
                check=False,  # 不抛出异常，我们需要检查返回码
                capture_output=True,
                text=True,
            )
        
        elapsed_time = time.time() - start_time
        
        # 解析输出
        stdout_lines = result.stdout.split('\n')
        stderr_lines = result.stderr.split('\n')
        
        # 提取统计信息
        success_count = 0
        fail_count = 0
        
        for line in stdout_lines:
            if "成功:" in line or "成功" in line:
                try:
                    parts = line.split(":")
                    if len(parts) > 1:
                        success_part = parts[1].strip().split("/")[0]
                        success_count = int(success_part)
                except:
                    pass
            if "失败:" in line or "失败" in line:
                try:
                    parts = line.split(":")
                    if len(parts) > 1:
                        fail_part = parts[1].strip().split("/")[0]
                        fail_count = int(fail_part)
                except:
                    pass
        
        print("\n" + "=" * 80)
        print("批量推理完成")
        print("=" * 80)
        print(f"返回码: {result.returncode}")
        print(f"总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        
        if result.returncode != 0:
            print(f"\n错误输出 (最后50行):")
            for line in stderr_lines[-50:]:
                if line.strip():
                    print(f"  {line}")
        
        # 检查输出完整性
        integrity_results = {}
        if args.check_integrity:
            print("\n" + "=" * 80)
            print("检查输出完整性...")
            print("=" * 80)
            
            # 找到所有输出目录
            output_base = Path(args.out_dir)
            if output_base.exists():
                traj_dirs = [d for d in output_base.iterdir() if d.is_dir()]
                
                print(f"找到 {len(traj_dirs)} 个输出目录")
                
                # 并行检查完整性
                integrity_lock = threading.Lock()
                integrity_stats = {
                    'total': 0,
                    'complete': 0,
                    'incomplete': 0,
                    'corrupted': 0,
                    'total_files': 0,
                    'errors': []
                }
                
                def check_one_traj(traj_dir):
                    traj_name = traj_dir.name
                    result = check_output_integrity(str(traj_dir), traj_name)
                    
                    with integrity_lock:
                        integrity_stats['total'] += 1
                        integrity_stats['total_files'] += result['file_count']
                        
                        if result['errors']:
                            integrity_stats['corrupted'] += 1
                            integrity_stats['errors'].extend(result['errors'])
                        elif result['has_images'] and result['has_depth'] and result['has_samples']:
                            integrity_stats['complete'] += 1
                        else:
                            integrity_stats['incomplete'] += 1
                        
                        integrity_results[traj_name] = result
                
                # 使用线程池并行检查
                with ThreadPoolExecutor(max_workers=min(10, len(traj_dirs))) as executor:
                    futures = [executor.submit(check_one_traj, traj_dir) for traj_dir in traj_dirs]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"检查完整性时出错: {e}")
                
                print(f"\n完整性检查结果:")
                print(f"  总traj数: {integrity_stats['total']}")
                print(f"  完整: {integrity_stats['complete']}")
                print(f"  不完整: {integrity_stats['incomplete']}")
                print(f"  损坏: {integrity_stats['corrupted']}")
                print(f"  总文件数: {integrity_stats['total_files']}")
                
                if integrity_stats['errors']:
                    print(f"\n错误详情 (前20个):")
                    for error in integrity_stats['errors'][:20]:
                        print(f"  - {error}")
                
                # 检查重复文件
                print("\n检查重复文件...")
                duplicate_count, duplicates = check_duplicate_files(args.out_dir)
                if duplicate_count > 0:
                    print(f"⚠️  发现 {duplicate_count} 个重复文件")
                    print(f"重复文件组数: {len(duplicates)}")
                    # 显示前5个重复组
                    for i, (hash_val, paths) in enumerate(list(duplicates.items())[:5]):
                        print(f"  重复组 {i+1}: {len(paths)} 个文件")
                        for path in paths[:3]:
                            print(f"    - {path}")
                else:
                    print("✅ 未发现重复文件")
        
        # 生成测试报告
        report = {
            'test_config': {
                'base_path': args.base_path,
                'out_dir': args.out_dir,
                'gpu_id': args.gpu_id,
                'max_trajs': args.max_trajs,
                'max_workers': args.max_workers,
                'frame_drop_rate': args.frame_drop_rate,
                'grid_size': args.grid_size,
            },
            'execution': {
                'return_code': result.returncode,
                'elapsed_time': elapsed_time,
                'success_count': success_count,
                'fail_count': fail_count,
            },
            'integrity': integrity_results if args.check_integrity else None,
        }
        
        # 保存报告
        report_path = os.path.join(args.out_dir, "stress_test_report.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n测试报告已保存到: {report_path}")
        
        # 如果指定了min_trajs_for_issue，检查是否达到了目标
        if args.min_trajs_for_issue:
            if os.path.exists(args.out_dir):
                traj_dirs = [d for d in Path(args.out_dir).iterdir() if d.is_dir()]
                completed_count = len(traj_dirs)
                
                if completed_count >= args.min_trajs_for_issue:
                    print(f"\n✅ 成功处理了 {completed_count} 个traj，达到了重现累积性问题的目标")
                else:
                    print(f"\n⚠️  只处理了 {completed_count} 个traj，未达到目标 {args.min_trajs_for_issue}")
                    print("   可能需要增加max_trajs或检查失败原因")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
        return False
    except Exception as e:
        print(f"\n❌ 测试执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="批量推理压力测试")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="数据集基础路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="GPU ID列表，用逗号分隔（如 0,1,2,3）",
    )
    parser.add_argument(
        "--max_trajs",
        type=int,
        default=100,
        help="最大traj数量（用于测试）。如果指定了--min_trajs_for_issue，会自动调整",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="最大工作进程数（默认等于GPU数量）",
    )
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=5,
        help="帧采样率",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="Grid size for keypoint sampling",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="检查输出文件完整性",
    )
    parser.add_argument(
        "--min_trajs_for_issue",
        type=int,
        default=None,
        help="最小traj数量（用于重现累积性问题，如conda竞争）。如果指定，将至少处理这么多traj",
    )
    parser.add_argument(
        "--monitor_interval",
        type=int,
        default=60,
        help="监控间隔（秒），用于定期检查运行状态",
    )
    parser.add_argument(
        "--save_progress",
        action="store_true",
        help="保存进度信息，支持断点续传",
    )
    
    args = parser.parse_args()
    
    # 如果指定了min_trajs_for_issue，确保max_trajs至少等于它
    if args.min_trajs_for_issue:
        if args.max_trajs is None or args.max_trajs < args.min_trajs_for_issue:
            print(f"⚠️  调整max_trajs从 {args.max_trajs} 到 {args.min_trajs_for_issue} 以重现累积性问题")
            args.max_trajs = args.min_trajs_for_issue
    
    success = run_stress_test(args)
    
    if success:
        print("\n✅ 压力测试完成")
        sys.exit(0)
    else:
        print("\n❌ 压力测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()

