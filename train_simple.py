#!/usr/bin/env python3
"""
简化的训练脚本 - 修复配置问题后的版本
"""
import os
import sys
import argparse

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from training.train import train_model, parse_arguments

def find_data_file():
    """自动查找数据文件"""
    data_file = "diabetes_prediction_dataset.csv"
    possible_paths = [
        f"/root/{data_file}",
        f"../{data_file}",
        f"./{data_file}",
        f"/home/ubuntu/{data_file}",
        f"~/{data_file}",
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return os.path.abspath(expanded_path)
    
    raise FileNotFoundError(f"找不到数据文件 {data_file}")

def create_args():
    """创建默认参数"""
    class Args:
        def __init__(self):
            # 自动查找数据文件
            try:
                self.data_path = find_data_file()
                print(f"✅ 找到数据文件: {self.data_path}")
            except FileNotFoundError:
                print("❌ 找不到数据文件，请确保 diabetes_prediction_dataset.csv 在正确位置")
                sys.exit(1)
            
            # 模型参数
            self.hidden_sizes = [64, 32, 16]
            self.dropout_rate = 0.2
            self.activation = 'relu'
            
            # 训练参数
            self.epochs = 50  # 减少轮数用于快速测试
            self.batch_size = 32
            self.learning_rate = 0.001
            self.early_stopping_patience = 10
            
            # 设备参数
            self.device = 'CPU'
            self.device_id = 0
            
            # 保存参数
            self.save_path = './checkpoints/diabetes_model.ckpt'
            self.log_dir = './logs'
            
            # 其他参数
            self.seed = 42
            self.eval_interval = 5  # 减少验证间隔
    
    return Args()

def main():
    """主函数"""
    print("🚀 开始简化训练流程...")
    
    try:
        # 创建参数
        args = create_args()
        
        # 开始训练
        model, val_metrics, test_metrics = train_model(args)
        
        print("\n✅ 训练成功完成!")
        print(f"验证集准确率: {val_metrics['accuracy']:.4f}")
        print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 