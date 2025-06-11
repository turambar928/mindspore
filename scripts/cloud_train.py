#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华为云ECS MindSpore训练脚本
优化了云端资源使用和监控
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mindspore as ms
from mindspore import context
import pandas as pd

from config.model_config import ModelConfig, DataConfig, TrainingConfig
from data.data_processor import DiabetesDataProcessor
from model.diabetes_net import DiabetesNet
from model.model_utils import DiabetesTrainer

def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def monitor_system_resources():
    """监控系统资源使用情况"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100
        }
    except ImportError:
        return None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='华为云MindSpore训练')
    
    # 数据参数
    parser.add_argument('--data_file', type=str, 
                        default='../diabetes_prediction_dataset.csv',
                        help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    
    # 模型参数
    parser.add_argument('--hidden_dims', type=str, default='256,128,64',
                        help='隐藏层维度，逗号分隔')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='CPU', 
                        choices=['CPU', 'GPU', 'Ascend'], help='设备类型')
    parser.add_argument('--device_id', type=int, default=0, help='设备ID')
    
    # 其他参数
    parser.add_argument('--save_model', action='store_true', help='是否保存模型')
    parser.add_argument('--log_interval', type=int, default=100, help='日志打印间隔')
    
    return parser.parse_args()

def setup_mindspore_context(device, device_id):
    """设置MindSpore运行环境"""
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=device,
        device_id=device_id,
        save_graphs=False
    )
    
    if device == 'GPU':
        context.set_context(enable_graph_kernel=True)
    elif device == 'Ascend':
        context.set_context(
            enable_graph_kernel=False,
            max_call_depth=10000
        )

def save_training_results(output_dir, results, model_config, training_config):
    """保存训练结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练结果
    results_file = os.path.join(output_dir, 'training_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'training_results': results,
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False, default=str)
    
    return results_file

def main():
    """主训练函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    logger.info("开始华为云MindSpore训练")
    logger.info(f"训练参数: {vars(args)}")
    
    try:
        # 设置MindSpore环境
        setup_mindspore_context(args.device, args.device_id)
        logger.info(f"MindSpore环境设置完成: {args.device}:{args.device_id}")
        
        # 监控系统资源
        system_info = monitor_system_resources()
        if system_info:
            logger.info(f"系统资源: CPU={system_info['cpu_percent']:.1f}%, "
                       f"内存={system_info['memory_percent']:.1f}%, "
                       f"磁盘={system_info['disk_percent']:.1f}%")
        
        # 创建配置
        hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
        
        model_config = ModelConfig(
            input_dim=8,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout_rate=args.dropout_rate,
            use_batch_norm=True
        )
        
        data_config = DataConfig(
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            normalize_features=True
        )
        
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            patience=args.patience,
            device=args.device.lower()
        )
        
        logger.info(f"模型配置: 隐藏层维度={hidden_dims}, dropout={args.dropout_rate}")
        logger.info(f"训练配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        
        # 加载数据
        logger.info("开始加载数据...")
        if not os.path.exists(args.data_file):
            raise FileNotFoundError(f"数据文件不存在: {args.data_file}")
        
        processor = DiabetesDataProcessor(data_config)
        df = pd.read_csv(args.data_file)
        
        train_dataset, val_dataset, test_dataset = processor.prepare_datasets(df)
        logger.info(f"数据加载完成 - 训练集: {len(train_dataset)}, "
                   f"验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
        
        # 创建模型和训练器
        logger.info("创建模型...")
        model = DiabetesNet(model_config)
        trainer = DiabetesTrainer(model, model_config, training_config)
        
        # 开始训练
        logger.info("开始训练...")
        start_time = time.time()
        
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        training_time = time.time() - start_time
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 模型评估
        logger.info("开始模型评估...")
        test_metrics = trainer.evaluate(test_dataset)
        
        logger.info("=== 训练结果 ===")
        logger.info(f"测试集准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"测试集精确率: {test_metrics['precision']:.4f}")
        logger.info(f"测试集召回率: {test_metrics['recall']:.4f}")
        logger.info(f"测试集F1分数: {test_metrics['f1_score']:.4f}")
        logger.info(f"训练时间: {training_time:.2f}秒")
        
        # 保存结果
        results = {
            'test_metrics': test_metrics,
            'training_history': history,
            'training_time': training_time,
            'system_info': system_info
        }
        
        results_file = save_training_results(
            args.output_dir, results, model_config, training_config
        )
        logger.info(f"训练结果已保存: {results_file}")
        
        # 保存模型
        if args.save_model:
            model_file = os.path.join(args.output_dir, 'diabetes_model.ckpt')
            ms.save_checkpoint(model, model_file)
            logger.info(f"模型已保存: {model_file}")
        
        logger.info("训练任务完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 