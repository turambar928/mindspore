"""
MindSpore糖尿病预测项目快速启动脚本
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        "mindspore", "numpy", "pandas", "scikit-learn", 
        "flask", "matplotlib", "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖包: {missing_packages}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_data():
    """检查数据文件"""
    print("\n📊 检查数据文件...")
    
    data_path = "../diabetes_prediction_dataset.csv"
    if os.path.exists(data_path):
        print(f"✅ 数据文件存在: {data_path}")
        return True
    else:
        print(f"❌ 数据文件不存在: {data_path}")
        print("请确保数据文件在正确位置")
        return False

def create_directories():
    """创建必要目录"""
    print("\n📁 创建目录结构...")
    
    directories = [
        "checkpoints", "logs", "data", "evaluation_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/")
    
    print("✅ 目录结构创建完成")

def train_model():
    """训练模型"""
    print("\n🏋️ 开始训练模型...")
    
    try:
        # 检查训练脚本
        train_script = "training/train.py"
        if not os.path.exists(train_script):
            print(f"❌ 训练脚本不存在: {train_script}")
            return False
        
        # 运行训练
        cmd = [
            sys.executable, train_script,
            "--data_path", "../diabetes_prediction_dataset.csv",
            "--epochs", "50",
            "--batch_size", "32",
            "--device", "CPU"
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 模型训练完成")
            return True
        else:
            print(f"❌ 训练失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 训练过程出错: {str(e)}")
        return False

def start_api_service():
    """启动API服务"""
    print("\n🌐 启动API服务...")
    
    try:
        api_script = "serving/mindspore_api.py"
        if not os.path.exists(api_script):
            print(f"❌ API脚本不存在: {api_script}")
            return False
        
        print("启动MindSpore API服务...")
        print("服务地址: http://localhost:8000")
        print("按Ctrl+C停止服务")
        
        # 启动服务
        subprocess.run([sys.executable, api_script])
        
    except KeyboardInterrupt:
        print("\n⚠️ 服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {str(e)}")

def quick_test():
    """快速测试"""
    print("\n🧪 运行快速测试...")
    
    try:
        # 测试模型创建
        print("测试模型创建...")
        import mindspore as ms
        ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
        
        from model.diabetes_net import DiabetesNet
        from config.model_config import ModelConfig
        
        config = ModelConfig()
        model = DiabetesNet(config)
        print("✅ 模型创建成功")
        
        # 测试数据处理
        print("测试数据处理...")
        from data.data_processor import DiabetesDataProcessor
        
        processor = DiabetesDataProcessor()
        sample_data = {
            "age": 45,
            "gender": "Male",
            "bmi": 28.5,
            "HbA1c_level": 6.5,
            "blood_glucose_level": 140,
            "smoking_history": "former",
            "hypertension": 1,
            "heart_disease": 0
        }
        
        features = processor.preprocess_single_sample(sample_data)
        print("✅ 数据处理成功")
        
        print("✅ 所有测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def show_menu():
    """显示菜单"""
    print("\n🧠 MindSpore糖尿病预测项目")
    print("=" * 40)
    print("1. 检查环境")
    print("2. 训练模型")
    print("3. 启动API服务")
    print("4. 运行快速测试")
    print("5. 完整部署流程")
    print("0. 退出")
    print("=" * 40)

def full_deployment():
    """完整部署流程"""
    print("\n🚀 开始完整部署流程")
    print("=" * 50)
    
    steps = [
        ("检查依赖", check_dependencies),
        ("检查数据", check_data),
        ("创建目录", create_directories),
        ("快速测试", quick_test),
        ("训练模型", train_model)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 步骤: {step_name}")
        if not step_func():
            print(f"❌ 步骤失败: {step_name}")
            return False
        time.sleep(1)
    
    print("\n🎉 完整部署流程完成!")
    print("现在可以启动API服务了")
    
    # 询问是否启动服务
    response = input("\n是否立即启动API服务? (y/n): ")
    if response.lower() == 'y':
        start_api_service()
    
    return True

def main():
    """主函数"""
    while True:
        show_menu()
        
        try:
            choice = input("请选择操作 (0-5): ").strip()
            
            if choice == "0":
                print("👋 再见!")
                break
            elif choice == "1":
                check_dependencies()
                check_data()
            elif choice == "2":
                train_model()
            elif choice == "3":
                start_api_service()
            elif choice == "4":
                quick_test()
            elif choice == "5":
                full_deployment()
            else:
                print("❌ 无效选择，请重新输入")
        
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
        
        input("\n按Enter键继续...")

if __name__ == "__main__":
    print("🧠 MindSpore糖尿病预测项目快速启动")
    print("=" * 50)
    main() 