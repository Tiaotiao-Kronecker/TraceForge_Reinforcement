#!/usr/bin/env python3
"""
批量推理 traj_group0 下的所有 traj 目录

支持的数据格式：
1. 旧格式: traj_group0/trajX/cam0/images0 或 traj_group0/trajX/images0
2. 新格式: bridge_depth/{数字目录}/images0 (如 /usr/data/dataset/opt/dataset_temp/bridge_depth/00000/images0)

用法:
    conda run -n traceforge python batch_infer.py \
        --base_path /home/user/dataset/bridgeV2/datacol2_toykitchen1/pnp_push_sweep/01/2023-07-03_14-41-20/raw/traj_group0 \
        --out_dir ./output_traj_group0 \
        --use_all_trajectories \
        --skip_existing \
        --frame_drop_rate 5
    
    或用于 bridge_depth 格式:
    conda run -n traceforge python batch_infer.py \
        --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
        --out_dir ./output_bridge_depth \
        --use_all_trajectories \
        --skip_existing \
        --frame_drop_rate 5
    
    测试模式（处理少量数据并估算时间）:
    conda run -n traceforge python batch_infer.py \
        --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
        --out_dir ./output_bridge_depth \
        --test_mode \
        --gpu_id 0
    
    指定GPU并行处理:
    conda run -n traceforge python batch_infer.py \
        --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
        --out_dir ./output_bridge_depth \
        --gpu_id 0,1,2,3 \
        --max_trajs 10
    
    串行处理（禁用并行）:
    conda run -n traceforge python batch_infer.py \
        --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
        --out_dir ./output_bridge_depth \
        --gpu_id 0,1,2,3 \
        --max_workers 1 \
        --max_trajs 10
"""

import os
import sys
import argparse

# 确保项目根目录在 sys.path 中，以便预加载模型时能正确导入 models、utils 等
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def find_valid_traj_dirs(base_path):
    """
    找到所有有效的traj目录，适配三种数据格式：
    格式1: trajX/cam0/images0 (如 traj0, traj1) - 旧格式
    格式2: trajX/images0 (如 traj2, traj3, ...) - 旧格式
    格式3: {数字目录}/images0 (如 00000, 00001, 01625) - bridge_depth格式
    所有格式都需要 depth_images0 存在
    """
    valid_trajs = []
    base_path = Path(base_path)
    
    # 获取所有子目录
    all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for traj_dir in sorted(all_dirs):
        depth_images0 = traj_dir / "depth_images0"
        if not depth_images0.exists():
            continue
        
        # 检查是否有深度图像文件
        has_depth = any(f.suffix.lower() in ['.png', '.jpg', '.jpeg'] 
                       for f in depth_images0.iterdir() if f.is_file())
        if not has_depth:
            continue
        
        # 尝试格式1: trajX/cam0/images0 (旧格式)
        cam0_images0 = traj_dir / "cam0" / "images0"
        video_path = None
        
        if cam0_images0.exists():
            # 检查是否有RGB图像文件
            has_images = any(f.suffix.lower() in ['.png', '.jpg', '.jpeg'] 
                           for f in cam0_images0.iterdir() if f.is_file())
            if has_images:
                video_path = cam0_images0
        else:
            # 尝试格式2和3: trajX/images0 或 {数字目录}/images0
            images0 = traj_dir / "images0"
            if images0.exists():
                # 检查是否有RGB图像文件
                has_images = any(f.suffix.lower() in ['.png', '.jpg', '.jpeg'] 
                               for f in images0.iterdir() if f.is_file())
                if has_images:
                    video_path = images0
        
        if video_path is not None:
            valid_trajs.append({
                'traj_name': traj_dir.name,
                'video_path': str(video_path),
                'depth_path': str(depth_images0),
            })
    
    return valid_trajs

def process_single_traj(traj_info, args, infer_script, gpu_ids, traj_index, total_trajs, print_lock):
    """
    处理单个traj的函数，用于并行执行
    
    Args:
        traj_info: traj信息字典
        args: 命令行参数
        infer_script: infer.py脚本路径
        gpu_ids: GPU ID列表
        traj_index: 当前traj的索引（从1开始）
        total_trajs: 总traj数量
        print_lock: 打印锁，用于同步输出
    
    Returns:
        (success, traj_name, elapsed_time, error_msg)
    """
    traj_name = traj_info['traj_name']
    video_path = traj_info['video_path']
    depth_path = traj_info['depth_path']
    
    traj_out_dir = os.path.join(args.out_dir, traj_name)
    
    # 分配GPU：如果指定了多个GPU，轮询分配
    if gpu_ids and isinstance(gpu_ids[0], int):
        gpu_id = gpu_ids[(traj_index - 1) % len(gpu_ids)]
        device = f"cuda:{gpu_id}"
    elif gpu_ids:
        device = gpu_ids[0]
    else:
        device = "cuda"
    
    # 打印开始信息（使用锁保证输出不混乱）
    with print_lock:
        print(f"\n{'='*80}")
        print(f"[{traj_index}/{total_trajs}] 开始处理 {traj_name}")
        print(f"{'='*80}")
        print(f"  Video: {video_path}")
        print(f"  Depth: {depth_path}")
        print(f"  Output: {traj_out_dir}")
        print(f"  Device: {device}")
    
    # 构建命令
    # 直接使用conda环境的Python解释器，避免conda run的插件冲突问题
    # 优先使用用户目录下的conda环境
    conda_env_python = "/home/wangchen/.conda/envs/traceforge/bin/python"
    if not os.path.exists(conda_env_python):
        # 如果不存在，尝试系统conda路径
        conda_env_python = "/usr/local/miniconda3/envs/traceforge/bin/python"
    
    # 如果还是不存在，回退到conda run（但添加--no-plugins选项）
    if not os.path.exists(conda_env_python):
        cmd = [
            "conda", "run", "--no-plugins", "-n", "traceforge", "python", infer_script,
            "--use_all_trajectories",
            "--frame_drop_rate", str(args.frame_drop_rate),
            "--out_dir", traj_out_dir,
            "--video_path", video_path,
            "--depth_path", depth_path,
            "--device", device,
        ]
    else:
        # 直接使用Python解释器（推荐方式）
        cmd = [
            conda_env_python, infer_script,
            "--use_all_trajectories",
            "--frame_drop_rate", str(args.frame_drop_rate),
            "--out_dir", traj_out_dir,
            "--video_path", video_path,
            "--depth_path", depth_path,
            "--device", device,
        ]
    
    if args.skip_existing:
        cmd.append("--skip_existing")
    
    if args.grid_size is not None:
        cmd.extend(["--grid_size", str(args.grid_size)])
    
    # 执行推理
    traj_start_time = time.time()
    try:
        # 使用subprocess.run执行，capture_output=True避免输出混乱
        # 设置环境变量，确保conda环境正确激活（如果使用直接Python路径）
        env = os.environ.copy()
        if 'conda_env_python' in locals() and conda_env_python and os.path.exists(conda_env_python):
            # 添加conda环境的bin目录到PATH
            conda_env_bin = os.path.dirname(conda_env_python)
            env['PATH'] = conda_env_bin + os.pathsep + env.get('PATH', '')
        # 确保子进程（infer.py）能找到项目根目录的 models、utils 等模块
        env['PYTHONPATH'] = _project_root + (os.pathsep + env.get('PYTHONPATH', '')) if env.get('PYTHONPATH') else _project_root
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        traj_elapsed = time.time() - traj_start_time
        
        # 检查输出目录是否存在且有内容
        # 注意：infer.py会在out_dir下创建video_name子目录（即images0）
        # 所以实际输出路径是 traj_out_dir/images0/
        expected_output_dir = os.path.join(traj_out_dir, os.path.basename(video_path.rstrip("/")))
        output_exists = os.path.exists(expected_output_dir)
        if output_exists:
            # 检查是否有实际文件（排除空目录）
            has_content = any(
                os.path.isfile(os.path.join(root, f))
                for root, dirs, files in os.walk(expected_output_dir)
                for f in files
            )
            # 统计文件数量
            file_count = sum(
                len(files)
                for root, dirs, files in os.walk(expected_output_dir)
            )
        else:
            has_content = False
            file_count = 0
        
        # 打印成功信息
        with print_lock:
            if has_content:
                status_icon = "✅"
                print(f"{status_icon} [{traj_index}/{total_trajs}] {traj_name} 处理完成 (耗时: {traj_elapsed:.2f}秒, GPU: {device}, 文件数: {file_count})")
            else:
                status_icon = "⚠️"
                print(f"{status_icon} [{traj_index}/{total_trajs}] {traj_name} 处理完成但输出为空 (耗时: {traj_elapsed:.2f}秒, GPU: {device})")
                print(f"   预期输出目录: {expected_output_dir}")
                print(f"   目录存在: {output_exists}")
                if result.returncode != 0:
                    print(f"   返回码: {result.returncode}")
                if result.stdout:
                    print(f"   STDOUT最后500字符: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"   STDERR最后500字符: {result.stderr[-500:]}")
        
        return (True, traj_name, traj_elapsed, None)
    except subprocess.CalledProcessError as e:
        traj_elapsed = time.time() - traj_start_time
        error_msg = f"返回码: {e.returncode}"
        
        with print_lock:
            print(f"❌ [{traj_index}/{total_trajs}] {traj_name} 处理失败 (耗时: {traj_elapsed:.2f}秒, GPU: {device}): {error_msg}")
            if e.stdout:
                stdout_lines = e.stdout.split('\n')
                print(f"   STDOUT (最后20行):")
                for line in stdout_lines[-20:]:
                    if line.strip():
                        print(f"     {line}")
            if e.stderr:
                stderr_lines = e.stderr.split('\n')
                print(f"   STDERR (最后30行):")
                for line in stderr_lines[-30:]:
                    if line.strip():
                        print(f"     {line}")
                # 查找关键错误信息
                error_keywords = ['CUDA', 'RuntimeError', 'illegal memory', 'out of memory', 'OOM', 'AssertionError', 'Traceback']
                for keyword in error_keywords:
                    matching_lines = [line for line in stderr_lines if keyword.lower() in line.lower()]
                    if matching_lines:
                        print(f"   ⚠️  发现关键词 '{keyword}':")
                        for line in matching_lines[:5]:  # 最多显示5行
                            print(f"     {line}")
        
        return (False, traj_name, traj_elapsed, error_msg)
    except Exception as e:
        traj_elapsed = time.time() - traj_start_time
        error_msg = str(e)
        
        with print_lock:
            print(f"❌ [{traj_index}/{total_trajs}] {traj_name} 处理异常 (耗时: {traj_elapsed:.2f}秒): {error_msg}")
        
        return (False, traj_name, traj_elapsed, error_msg)

def preload_models():
    """
    预先加载模型到缓存，避免每个子进程都尝试下载
    """
    print("=" * 80)
    print("预加载模型到缓存...")
    print("=" * 80)
    
    try:
        # 预加载VGGT4模型
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
        print("正在预加载 VGGT4Track (Yuxihenry/SpatialTrackerV2_Front)...")
        
        # 添加重试机制（与video_depth_pose_utils.py中的逻辑一致）
        import time
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
                print("✅ VGGT4Track 预加载成功")
                del model  # 释放内存
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⚠️  预加载失败 (尝试 {attempt + 1}/{max_retries})，等待 {wait_time} 秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    raise
        
        # 预加载MoGe模型（如果需要）
        try:
            from models.SpaTrackV2.models.SpaTrack import MoGeModel
            print("正在预加载 MoGeModel (Ruicheng/moge-vitl)...")
            moge_model = MoGeModel.from_pretrained('Ruicheng/moge-vitl')
            print("✅ MoGeModel 预加载成功")
            del moge_model
        except Exception as e:
            print(f"⚠️  MoGeModel 预加载失败（可能不需要）: {e}")
        
        print("=" * 80)
        print("模型预加载完成！")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"❌ 模型预加载失败: {e}")
        print("⚠️  警告: 将继续运行，但每个子进程可能会尝试下载模型")
        print("   如果遇到SSL错误，请检查网络连接或手动下载模型")
        return False

def main():
    parser = argparse.ArgumentParser(description="批量推理 traj_group0 下的所有 traj")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="数据集基础路径（支持 traj_group0 或 bridge_depth 格式）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./output_traj_group0",
        help="输出目录",
    )
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        help="使用所有轨迹",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过已存在的输出",
    )
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=5,
        help="帧采样率",
    )
    parser.add_argument(
        "--max_trajs",
        type=int,
        default=None,
        help="最大处理traj数量（用于测试）",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="指定GPU序号，可以是单个数字（如 0）或多个用逗号分隔（如 0,1,2）。如果指定多个GPU，任务会轮询分配。默认使用所有可用GPU",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="测试模式：处理少量traj并估算全部时间",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="禁用并行处理，使用串行模式",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="最大并行工作进程数。默认等于GPU数量。如果设置为1，则串行处理",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="Grid size for uniform keypoint sampling (grid_size x grid_size points per frame). Default is 20 (400 points).",
    )
    
    args = parser.parse_args()
    
    # 处理GPU ID参数
    gpu_ids = []
    if args.gpu_id:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_id.split(',')]
            # 验证GPU ID有效性
            import torch
            if torch.cuda.is_available():
                max_gpu = torch.cuda.device_count()
                invalid_gpus = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= max_gpu]
                if invalid_gpus:
                    print(f"❌ 警告: GPU {invalid_gpus} 不存在。可用GPU数量: {max_gpu}")
                    gpu_ids = [gpu for gpu in gpu_ids if 0 <= gpu < max_gpu]
                if not gpu_ids:
                    print(f"❌ 错误: 没有有效的GPU ID")
                    return
        except ValueError:
            print(f"❌ 错误: 无效的GPU ID格式: {args.gpu_id}")
            return
    else:
        # 如果没有指定，检查是否有CUDA可用
        try:
            import torch
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            else:
                print("⚠️  警告: 未检测到CUDA，将使用CPU")
                gpu_ids = ['cpu']
        except ImportError:
            print("⚠️  警告: 无法导入torch，将使用默认设备")
            gpu_ids = ['cuda']
    
    # 测试模式：如果指定了test_mode但没有指定max_trajs，默认处理5个
    if args.test_mode and args.max_trajs is None:
        args.max_trajs = 5
        print(f"🧪 测试模式：将处理前 {args.max_trajs} 个traj进行测试")
    
    # 找到所有有效的traj目录
    valid_trajs = find_valid_traj_dirs(args.base_path)
    
    print("=" * 80)
    print("批量推理 traj_group0")
    print("=" * 80)
    print(f"\n📁 基础路径: {args.base_path}")
    print(f"📊 找到 {len(valid_trajs)} 个有效的traj目录")
    
    # 记录总traj数量（用于时间估算）
    total_traj_count = len(valid_trajs)
    
    if args.max_trajs:
        valid_trajs = valid_trajs[:args.max_trajs]
        print(f"⚠️  限制处理前 {len(valid_trajs)} 个traj（用于测试）")
    
    # 显示GPU配置
    if gpu_ids and isinstance(gpu_ids[0], int):
        if len(gpu_ids) == 1:
            print(f"🎮 使用GPU: {gpu_ids[0]}")
        else:
            print(f"🎮 使用GPU: {gpu_ids} (并行处理)")
    elif gpu_ids:
        print(f"🎮 使用设备: {gpu_ids[0]}")
    else:
        print(f"🎮 使用设备: cuda (默认)")
    
    if not valid_trajs:
        print("❌ 没有找到有效的traj目录")
        return
    
    # 预加载模型到缓存（避免每个子进程都尝试下载）
    print("\n" + "=" * 80)
    print("步骤1: 预加载模型到缓存")
    print("=" * 80)
    preload_success = preload_models()
    if not preload_success:
        print("⚠️  警告: 模型预加载失败，将继续运行但可能遇到SSL错误")
        # 非交互式环境：自动继续；交互式环境：询问用户
        try:
            response = input("是否继续？(y/n，默认y): ").strip().lower()
            if response and response != 'y':
                print("已取消")
                return
        except (EOFError, KeyboardInterrupt):
            # 非交互式环境（如后台运行），自动继续
            print("(非交互式环境，自动继续)")
    
    # 构建infer.py命令的基础参数（不包含 --out_dir，因为每个 traj 需要独立的输出目录）
    infer_script = os.path.join(os.path.dirname(__file__), "infer.py")
    
    # 确定是否并行处理以及并行度
    use_parallel = not args.no_parallel  # 默认启用并行
    if args.max_workers == 1:
        use_parallel = False
        print("📌 串行处理模式（max_workers=1）")
    elif args.max_workers is None:
        # 默认并行度等于GPU数量
        if gpu_ids and isinstance(gpu_ids[0], int):
            max_workers = len(gpu_ids)
        else:
            max_workers = 1
    else:
        max_workers = args.max_workers
    
    if use_parallel and max_workers > 1:
        print(f"🚀 并行处理模式: 最多同时处理 {max_workers} 个traj")
    else:
        print(f"📌 串行处理模式")
        max_workers = 1
    
    # 处理每个traj
    success_count = 0
    fail_count = 0
    traj_times = []  # 记录每个traj的处理时间
    start_time = time.time()  # 总开始时间
    print_lock = threading.Lock()  # 用于同步打印输出
    
    if use_parallel and max_workers > 1:
        # 并行处理模式
        print(f"\n开始并行处理 {len(valid_trajs)} 个traj...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_traj = {
                executor.submit(
                    process_single_traj,
                    traj_info,
                    args,
                    infer_script,
                    gpu_ids,
                    i + 1,
                    len(valid_trajs),
                    print_lock
                ): (i + 1, traj_info['traj_name'])
                for i, traj_info in enumerate(valid_trajs)
            }
            
            # 处理完成的任务
            try:
                for future in as_completed(future_to_traj):
                    traj_index, traj_name = future_to_traj[future]
                    try:
                        success, name, elapsed, error = future.result()
                        traj_times.append(elapsed)
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        fail_count += 1
                        with print_lock:
                            print(f"❌ [{traj_index}/{len(valid_trajs)}] {traj_name} 处理异常: {e}")
            except KeyboardInterrupt:
                print(f"\n⚠️  用户中断，正在等待当前任务完成...")
                # 取消未完成的任务
                for future in future_to_traj:
                    future.cancel()
                print(f"⚠️  已取消未开始的任务")
    else:
        # 串行处理模式（保持原有逻辑）
        for i, traj_info in enumerate(valid_trajs, 1):
            success, traj_name, elapsed, error = process_single_traj(
                traj_info, args, infer_script, gpu_ids, i, len(valid_trajs), print_lock
            )
            traj_times.append(elapsed)
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("批量推理完成")
    print(f"{'='*80}")
    print(f"  成功: {success_count}/{len(valid_trajs)}")
    print(f"  失败: {fail_count}/{len(valid_trajs)}")
    print(f"  输出目录: {args.out_dir}")
    
    # 显示时间统计
    if traj_times:
        avg_time = sum(traj_times) / len(traj_times)
        min_time = min(traj_times)
        max_time = max(traj_times)
        
        print(f"\n⏱️  时间统计:")
        print(f"  总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
        print(f"  平均每个traj: {avg_time:.2f}秒")
        print(f"  最快: {min_time:.2f}秒")
        print(f"  最慢: {max_time:.2f}秒")
        
        # 如果是测试模式，估算全部数据集的时间
        if args.test_mode or args.max_trajs:
            if success_count > 0:
                estimated_total_time = avg_time * total_traj_count
                estimated_hours = estimated_total_time / 3600
                estimated_days = estimated_hours / 24
                
                print(f"\n📊 时间估算（基于当前测试结果）:")
                print(f"  总traj数量: {total_traj_count}")
                print(f"  预计总耗时: {estimated_total_time:.2f}秒 ({estimated_total_time/60:.2f}分钟 / {estimated_hours:.2f}小时 / {estimated_days:.2f}天)")
                
                # 如果使用多个GPU，考虑并行加速
                if gpu_ids and isinstance(gpu_ids[0], int) and len(gpu_ids) > 1:
                    parallel_estimated_time = estimated_total_time / len(gpu_ids)
                    parallel_hours = parallel_estimated_time / 3600
                    parallel_days = parallel_hours / 24
                    print(f"  使用 {len(gpu_ids)} 个GPU并行预计耗时: {parallel_estimated_time:.2f}秒 ({parallel_estimated_time/60:.2f}分钟 / {parallel_hours:.2f}小时 / {parallel_days:.2f}天)")

if __name__ == "__main__":
    main()

