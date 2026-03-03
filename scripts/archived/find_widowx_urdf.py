#!/usr/bin/env python3
"""
查找WidowX 250机械臂的URDF文件
支持多种查找方式：
1. BridgeData V2 GitHub仓库
2. Interbotix官方ROS包
3. 本地ROS工作空间
4. 系统ROS包路径
"""

import os
import subprocess
import sys
from pathlib import Path

def find_in_ros_packages():
    """在ROS包中查找"""
    print("=" * 80)
    print("方法1: 在ROS包中查找")
    print("=" * 80)
    
    # 尝试使用rospack查找
    packages_to_check = [
        "interbotix_wx250_description",
        "interbotix_wx250s_description", 
        "interbotix_xsarm_description",
        "widowx_arm_description",
        "widowx_250_description"
    ]
    
    found_packages = []
    for pkg in packages_to_check:
        try:
            result = subprocess.run(
                ["rospack", "find", pkg],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                pkg_path = result.stdout.strip()
                found_packages.append((pkg, pkg_path))
                print(f"✅ 找到ROS包: {pkg}")
                print(f"   路径: {pkg_path}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # 在找到的包中查找URDF文件
    urdf_files = []
    for pkg_name, pkg_path in found_packages:
        urdf_dirs = [
            os.path.join(pkg_path, "urdf"),
            os.path.join(pkg_path, "urdf", "wx250"),
            os.path.join(pkg_path, "urdf", "wx250s"),
            os.path.join(pkg_path, "description", "urdf"),
        ]
        
        for urdf_dir in urdf_dirs:
            if os.path.exists(urdf_dir):
                for file in os.listdir(urdf_dir):
                    if file.endswith(('.urdf', '.xacro')):
                        full_path = os.path.join(urdf_dir, file)
                        urdf_files.append((pkg_name, full_path))
                        print(f"   找到URDF文件: {file}")
                        print(f"   完整路径: {full_path}")
    
    return urdf_files

def find_in_ros_workspace():
    """在ROS工作空间中查找"""
    print("\n" + "=" * 80)
    print("方法2: 在ROS工作空间中查找")
    print("=" * 80)
    
    # 常见的ROS工作空间路径
    workspace_paths = [
        os.path.expanduser("~/catkin_ws"),
        os.path.expanduser("~/ros_ws"),
        os.path.expanduser("~/workspace"),
        "/opt/ros",
    ]
    
    urdf_files = []
    for ws_path in workspace_paths:
        if not os.path.exists(ws_path):
            continue
            
        print(f"\n检查工作空间: {ws_path}")
        try:
            # 递归查找URDF文件
            for root, dirs, files in os.walk(ws_path):
                # 跳过隐藏目录和build目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'build']
                
                for file in files:
                    if 'widowx' in file.lower() and file.endswith(('.urdf', '.xacro')):
                        full_path = os.path.join(root, file)
                        urdf_files.append(("workspace", full_path))
                        print(f"✅ 找到: {full_path}")
        except PermissionError:
            print(f"   权限不足，跳过: {ws_path}")
    
    return urdf_files

def find_in_system_paths():
    """在系统路径中查找"""
    print("\n" + "=" * 80)
    print("方法3: 在系统路径中查找")
    print("=" * 80)
    
    system_paths = [
        "/opt/ros",
        "/usr/share/ros",
        "/usr/local/share/ros",
    ]
    
    urdf_files = []
    for sys_path in system_paths:
        if not os.path.exists(sys_path):
            continue
            
        print(f"\n检查系统路径: {sys_path}")
        try:
            for root, dirs, files in os.walk(sys_path):
                # 限制深度，避免搜索过深
                depth = root[len(sys_path):].count(os.sep)
                if depth > 3:
                    dirs[:] = []
                    continue
                
                for file in files:
                    if 'widowx' in file.lower() and file.endswith(('.urdf', '.xacro')):
                        full_path = os.path.join(root, file)
                        urdf_files.append(("system", full_path))
                        print(f"✅ 找到: {full_path}")
        except PermissionError:
            print(f"   权限不足，跳过: {sys_path}")
    
    return urdf_files

def check_bridgedata_repo():
    """检查BridgeData V2仓库信息"""
    print("\n" + "=" * 80)
    print("方法4: BridgeData V2 GitHub仓库信息")
    print("=" * 80)
    
    print("""
BridgeData V2 GitHub仓库:
- 主仓库: https://github.com/rail-berkeley/bridge_data_v2
- 可能包含URDF的位置:
  * assets/robots/
  * config/robots/
  * urdf/
  * dependencies/

建议手动检查:
1. 访问 https://github.com/rail-berkeley/bridge_data_v2
2. 搜索 "urdf" 或 "widowx"
3. 查看 README.md 或 setup 文档中的依赖说明
    """)

def check_interbotix_repo():
    """检查Interbotix官方仓库信息"""
    print("\n" + "=" * 80)
    print("方法5: Interbotix官方仓库信息")
    print("=" * 80)
    
    print("""
Interbotix官方GitHub仓库:
- 主仓库: https://github.com/Interbotix/interbotix_ros_manipulators
- URDF文件通常位于:
  * interbotix_ros_manipulators/interbotix_descriptions/urdf/
  * interbotix_ros_manipulators/interbotix_xs_sdk/urdf/
  
WidowX 250相关包:
- interbotix_wx250_description (ROS包名)
- 或 interbotix_xsarm_description (通用XS系列)

下载方式:
1. 克隆仓库:
   git clone https://github.com/Interbotix/interbotix_ros_manipulators.git
   
2. 查找URDF:
   find interbotix_ros_manipulators -name "*wx250*.urdf" -o -name "*wx250*.xacro"
    """)

def main():
    print("=" * 80)
    print("WidowX 250 机械臂 URDF 文件查找工具")
    print("=" * 80)
    
    all_urdf_files = []
    
    # 方法1: ROS包
    urdf_files = find_in_ros_packages()
    all_urdf_files.extend(urdf_files)
    
    # 方法2: ROS工作空间
    urdf_files = find_in_ros_workspace()
    all_urdf_files.extend(urdf_files)
    
    # 方法3: 系统路径
    urdf_files = find_in_system_paths()
    all_urdf_files.extend(urdf_files)
    
    # 方法4: BridgeData仓库信息
    check_bridgedata_repo()
    
    # 方法5: Interbotix仓库信息
    check_interbotix_repo()
    
    # 总结
    print("\n" + "=" * 80)
    print("查找结果总结")
    print("=" * 80)
    
    if all_urdf_files:
        print(f"\n✅ 找到 {len(all_urdf_files)} 个可能的URDF文件:\n")
        for i, (source, path) in enumerate(all_urdf_files, 1):
            print(f"{i}. [{source}] {path}")
        
        print("\n推荐使用:")
        # 优先选择ROS包中的文件
        ros_pkg_files = [f for s, f in all_urdf_files if s != "workspace" and s != "system"]
        if ros_pkg_files:
            print(f"  {ros_pkg_files[0]}")
        else:
            print(f"  {all_urdf_files[0][1]}")
    else:
        print("\n❌ 未在本地找到URDF文件")
        print("\n建议:")
        print("1. 安装Interbotix ROS包:")
        print("   sudo apt-get install ros-<distro>-interbotix-wx250-description")
        print("\n2. 或从GitHub克隆Interbotix仓库:")
        print("   git clone https://github.com/Interbotix/interbotix_ros_manipulators.git")
        print("\n3. 或检查BridgeData V2仓库:")
        print("   https://github.com/rail-berkeley/bridge_data_v2")

if __name__ == "__main__":
    main()

