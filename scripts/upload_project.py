#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上传MindSpore项目到华为云ECS服务器
"""

import os
import subprocess
import argparse
import tarfile
import tempfile

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='上传MindSpore项目到ECS服务器')
    parser.add_argument('--server_ip', type=str, required=True, help='ECS服务器IP地址')
    parser.add_argument('--username', type=str, default='root', help='服务器用户名')
    parser.add_argument('--key_file', type=str, help='SSH私钥文件路径')
    parser.add_argument('--password', type=str, help='服务器密码（如果不使用密钥）')
    parser.add_argument('--project_path', type=str, default='.', help='本地项目路径')
    parser.add_argument('--remote_path', type=str, default='~/mindspore_project', help='远程项目路径')
    
    return parser.parse_args()

def create_project_archive(project_path):
    """创建项目压缩包"""
    print("正在创建项目压缩包...")
    
    # 创建临时压缩文件
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        archive_path = tmp_file.name
    
    # 需要包含的文件和目录
    include_patterns = [
        'config/',
        'data/',
        'model/',
        'training/',
        'serving/',
        'scripts/',
        'requirements.txt',
        'README.md',
        'quick_start.py'
    ]
    
    # 需要排除的文件和目录
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '.git',
        '.DS_Store',
        '*.log',
        'output/',
        'logs/',
        '.vscode/',
        '.idea/'
    ]
    
    with tarfile.open(archive_path, 'w:gz') as tar:
        for pattern in include_patterns:
            full_path = os.path.join(project_path, pattern)
            if os.path.exists(full_path):
                tar.add(full_path, arcname=pattern)
                print(f"添加: {pattern}")
    
    print(f"项目压缩包创建完成: {archive_path}")
    return archive_path

def upload_to_server(archive_path, server_ip, username, remote_path, key_file=None, password=None):
    """上传文件到服务器"""
    print(f"正在上传到服务器 {server_ip}...")
    
    # 构建scp命令
    if key_file:
        scp_cmd = [
            'scp', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
            archive_path, f'{username}@{server_ip}:~/mindspore_project.tar.gz'
        ]
    else:
        # 如果没有密钥文件，需要使用sshpass
        scp_cmd = [
            'sshpass', '-p', password,
            'scp', '-o', 'StrictHostKeyChecking=no',
            archive_path, f'{username}@{server_ip}:~/mindspore_project.tar.gz'
        ]
    
    try:
        result = subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
        print("文件上传成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"上传失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def extract_on_server(server_ip, username, remote_path, key_file=None, password=None):
    """在服务器上解压项目"""
    print("正在服务器上解压项目...")
    
    # 构建SSH命令
    extract_commands = [
        f'mkdir -p {remote_path}',
        f'cd {remote_path}',
        'tar -xzf ~/mindspore_project.tar.gz',
        'rm ~/mindspore_project.tar.gz',
        'ls -la'
    ]
    
    ssh_command = ' && '.join(extract_commands)
    
    if key_file:
        ssh_cmd = [
            'ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
            f'{username}@{server_ip}', ssh_command
        ]
    else:
        ssh_cmd = [
            'sshpass', '-p', password,
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{username}@{server_ip}', ssh_command
        ]
    
    try:
        result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
        print("项目解压成功！")
        print("服务器文件列表:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"解压失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 1. 创建项目压缩包
        archive_path = create_project_archive(args.project_path)
        
        # 2. 上传到服务器
        if upload_to_server(archive_path, args.server_ip, args.username, 
                          args.remote_path, args.key_file, args.password):
            
            # 3. 在服务器上解压
            extract_on_server(args.server_ip, args.username, args.remote_path,
                             args.key_file, args.password)
            
            print("\n=== 项目部署完成 ===")
            print(f"项目已上传到: {args.server_ip}:{args.remote_path}")
            print("\n接下来的步骤:")
            print(f"1. SSH登录服务器: ssh {args.username}@{args.server_ip}")
            print(f"2. 进入项目目录: cd {args.remote_path}")
            print("3. 运行部署脚本: bash scripts/deploy_to_ecs.sh")
            print("4. 激活环境: source mindspore_env/bin/activate")
            print("5. 开始训练: python training/train.py")
        
        # 清理临时文件
        os.unlink(archive_path)
        
    except Exception as e:
        print(f"部署过程中发生错误: {e}")

if __name__ == "__main__":
    main() 