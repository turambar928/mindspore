"""
简化版MindSpore部署脚本
"""
import os
import json
import shutil
from datetime import datetime

def package_model():
    """打包模型文件"""
    print("📦 打包模型文件...")
    
    # 创建部署包目录
    deploy_dir = "mindspore_deploy_package"
    os.makedirs(deploy_dir, exist_ok=True)
    
    # 需要打包的文件
    files_to_package = [
        "config/",
        "model/",
        "data/",
        "serving/",
        "requirements.txt"
    ]
    
    for item in files_to_package:
        src = item
        dst = os.path.join(deploy_dir, item)
        
        if os.path.exists(src):
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"✅ 已打包: {item}")
        else:
            print(f"⚠️ 文件不存在: {item}")
    
    # 创建部署配置
    deploy_config = {
        "project_name": "mindspore-diabetes-prediction",
        "version": "1.0.0",
        "deploy_time": datetime.now().isoformat(),
        "python_version": "3.8+",
        "dependencies": [
            "mindspore>=2.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "flask>=2.3.0",
            "scikit-learn>=1.3.0"
        ]
    }
    
    with open(os.path.join(deploy_dir, "deploy_config.json"), "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 部署包已创建: {deploy_dir}/")

def create_startup_script():
    """创建启动脚本"""
    print("🚀 创建启动脚本...")
    
    startup_script = """#!/bin/bash
# MindSpore糖尿病预测服务启动脚本

echo "🧠 启动MindSpore糖尿病预测服务..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖包..."
python3 -c "import mindspore, numpy, pandas, flask" 2>/dev/null || {
    echo "❌ 缺少必要依赖包，请运行: pip install -r requirements.txt"
    exit 1
}

# 启动服务
echo "🌐 启动API服务..."
cd serving
python3 mindspore_api.py

echo "✅ 服务已启动"
"""
    
    with open("mindspore_deploy_package/start_service.sh", "w", encoding="utf-8") as f:
        f.write(startup_script)
    
    # 设置执行权限
    os.chmod("mindspore_deploy_package/start_service.sh", 0o755)
    
    print("✅ 启动脚本已创建")

def create_docker_config():
    """创建Docker配置"""
    print("🐳 创建Docker配置...")
    
    dockerfile = """FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "serving/mindspore_api.py"]
"""
    
    with open("mindspore_deploy_package/Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile)
    
    # Docker compose文件
    docker_compose = """version: '3.8'

services:
  mindspore-diabetes-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MINDSPORE_SERVING_ENABLE=1
    volumes:
      - ./checkpoints:/app/checkpoints
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    with open("mindspore_deploy_package/docker-compose.yml", "w", encoding="utf-8") as f:
        f.write(docker_compose)
    
    print("✅ Docker配置已创建")

def main():
    """主函数"""
    print("🧠 MindSpore简化部署工具")
    print("=" * 40)
    
    try:
        # 打包模型
        package_model()
        
        # 创建启动脚本
        create_startup_script()
        
        # 创建Docker配置
        create_docker_config()
        
        print("\n🎉 部署包创建完成!")
        print("📁 部署包位置: mindspore_deploy_package/")
        print("\n📋 部署步骤:")
        print("1. 将部署包上传到目标服务器")
        print("2. 安装依赖: pip install -r requirements.txt")
        print("3. 启动服务: ./start_service.sh")
        print("   或使用Docker: docker-compose up -d")
        
    except Exception as e:
        print(f"❌ 部署包创建失败: {str(e)}")

if __name__ == "__main__":
    main() 