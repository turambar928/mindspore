"""
MindSpore糖尿病预测模型训练脚本
"""
import os
import argparse
import time
from datetime import datetime
import mindspore as ms
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import ModelConfig, TrainingConfig, update_config_for_device
from data.data_processor import prepare_data_for_training
from model.diabetes_net import DiabetesNet, DiabetesLoss
from model.model_utils import create_trainer, save_model, model_summary, ValidationCallback

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MindSpore糖尿病预测模型训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                       default="./diabetes_prediction_dataset.csv",
                       help='训练数据路径')
    
    # 模型参数
    parser.add_argument('--hidden_sizes', type=int, nargs='+', 
                       default=[64, 32, 16],
                       help='隐藏层尺寸')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout率')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'sigmoid', 'tanh'],
                       help='激活函数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='早停耐心值')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='CPU',
                       choices=['CPU', 'GPU', 'Ascend'],
                       help='计算设备')
    parser.add_argument('--device_id', type=int, default=0,
                       help='设备ID')
    
    # 保存参数
    parser.add_argument('--save_path', type=str, 
                       default='./checkpoints/diabetes_model.ckpt',
                       help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='验证间隔轮数')
    
    return parser.parse_args()

def setup_environment(args):
    """设置环境"""
    # 设置随机种子
    ms.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置MindSpore上下文
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target=args.device,
        device_id=args.device_id
    )
    
    # 更新配置
    update_config_for_device(args.device)
    
    # 创建目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"🚀 环境设置完成")
    print(f"设备: {args.device} (ID: {args.device_id})")
    print(f"随机种子: {args.seed}")

def create_configs(args):
    """创建配置"""
    # 模型配置
    model_config = ModelConfig()
    model_config.hidden_sizes = args.hidden_sizes
    model_config.dropout_rate = args.dropout_rate
    model_config.activation = args.activation
    
    # 训练配置
    training_config = TrainingConfig()
    training_config.device_target = args.device
    training_config.device_id = args.device_id
    training_config.learning_rate = args.learning_rate
    training_config.batch_size = args.batch_size
    training_config.epochs = args.epochs
    training_config.early_stopping_patience = args.early_stopping_patience
    training_config.model_save_path = os.path.dirname(args.save_path)
    training_config.log_path = args.log_dir
    training_config.eval_interval = args.eval_interval
    
    return model_config, training_config

def train_model(args):
    """训练模型"""
    print("🏋️ 开始训练MindSpore糖尿病预测模型")
    print("=" * 60)
    
    # 设置环境
    setup_environment(args)
    
    # 创建配置
    model_config, training_config = create_configs(args)
    
    # 加载数据
    print("📊 加载和预处理数据...")
    train_dataset, val_dataset, test_dataset = prepare_data_for_training(
        data_path=args.data_path,
        batch_size=training_config.batch_size
    )
    
    print(f"训练集大小: {train_dataset.get_dataset_size()}")
    print(f"验证集大小: {val_dataset.get_dataset_size()}")
    print(f"测试集大小: {test_dataset.get_dataset_size()}")
    
    # 创建模型
    print("\n🧠 创建模型...")
    model = DiabetesNet(model_config)
    
    # 打印模型信息
    print_model_info(model)
    
    # 创建训练器
    print("\n⚙️ 创建训练器...")
    trainer, callbacks = create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config
    )
    
    # 开始训练
    print(f"\n🚀 开始训练 (共{training_config.epochs}轮)...")
    start_time = time.time()
    
    try:
        trainer.train(
            epoch=training_config.epochs,
            train_dataset=train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=False
        )
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n⏱️ 训练完成! 总耗时: {training_time:.2f}秒")
    
    # 保存最终模型
    print("\n💾 保存最终模型...")
    save_model(model, args.save_path, model_config)
    
    # 最终评估
    print("\n📊 最终模型评估...")
    from model.model_utils import evaluate_model
    
    # 验证集评估
    val_metrics = evaluate_model(model, val_dataset)
    print("验证集指标:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 测试集评估
    test_metrics = evaluate_model(model, test_dataset)
    print("\n测试集指标:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存训练报告
    save_training_report(args, model_config, training_config, 
                        val_metrics, test_metrics, training_time)
    
    print("\n🎉 训练流程完成!")
    return model, val_metrics, test_metrics

def print_model_info(model):
    """打印模型信息"""
    print("🧠 MindSpore糖尿病预测模型")
    print("-" * 40)
    print(f"输入维度: {model.input_size}")
    print(f"隐藏层: {model.hidden_sizes}")
    print(f"输出维度: {model.output_size}")
    print(f"Dropout率: {model.dropout_rate}")
    print(f"激活函数: {model.config.activation}")
    
    # 计算参数数量
    total_params = sum(p.size for p in model.trainable_params())
    print(f"可训练参数: {total_params:,}")

def save_training_report(args, model_config, training_config, 
                        val_metrics, test_metrics, training_time):
    """保存训练报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.log_dir, f"training_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MindSpore糖尿病预测模型训练报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("训练参数:\n")
        f.write(f"  数据路径: {args.data_path}\n")
        f.write(f"  训练轮数: {training_config.epochs}\n")
        f.write(f"  批大小: {training_config.batch_size}\n")
        f.write(f"  学习率: {training_config.learning_rate}\n")
        f.write(f"  设备: {training_config.device_target}\n")
        f.write(f"  训练时间: {training_time:.2f}秒\n\n")
        
        f.write("模型架构:\n")
        f.write(f"  隐藏层: {model_config.hidden_sizes}\n")
        f.write(f"  Dropout率: {model_config.dropout_rate}\n")
        f.write(f"  激活函数: {model_config.activation}\n\n")
        
        f.write("验证集指标:\n")
        for metric, value in val_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("测试集指标:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"📝 训练报告已保存: {report_path}")

def main():
    """主函数"""
    try:
        args = parse_arguments()
        
        # 检查数据文件
        if not os.path.exists(args.data_path):
            print(f"❌ 错误: 数据文件不存在: {args.data_path}")
            print("请确保数据文件路径正确")
            return
        
        # 开始训练
        model, val_metrics, test_metrics = train_model(args)
        
        print("\n✅ 训练成功完成!")
        print(f"模型已保存到: {args.save_path}")
        print(f"最佳验证准确率: {val_metrics['accuracy']:.4f}")
        print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 