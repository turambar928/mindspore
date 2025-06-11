#!/bin/bash
# ECS服务器部署脚本

echo "=== 华为云ECS MindSpore部署脚本 ==="

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python 3.9
sudo apt install -y python3.9 python3.9-venv python3.9-dev python3-pip
sudo ln -sf /usr/bin/python3.9 /usr/bin/python3
sudo ln -sf /usr/bin/pip3 /usr/bin/pip

# 安装系统依赖
sudo apt install -y build-essential cmake git wget curl

# 创建项目目录
mkdir -p ~/mindspore_project
cd ~/mindspore_project

# 创建虚拟环境
python3 -m venv mindspore_env
source mindspore_env/bin/activate

# 升级pip
pip install --upgrade pip

echo "=== 安装MindSpore依赖 ==="

# 安装MindSpore CPU版本（如需GPU版本需要安装CUDA）
pip install mindspore==2.3.1
pip install numpy pandas scikit-learn matplotlib seaborn
pip install flask flask-cors gunicorn
pip install jupyter tqdm requests psutil

echo "=== 项目部署完成 ==="
echo "请将项目文件上传到 ~/mindspore_project/ 目录"
echo "然后运行: source mindspore_env/bin/activate"
echo "开始训练: python training/train.py" 