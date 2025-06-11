"""
MindSpore糖尿病预测模型工具函数
"""
import os
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, dtype as mstype
from mindspore.train import Model
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import LossMonitor, TimeMonitor
import mindspore.dataset as ds

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import ModelConfig, TrainingConfig
from model.diabetes_net import DiabetesNet, DiabetesLoss, DiabetesMetrics

class ValidationCallback(Callback):
    """验证回调函数"""
    
    def __init__(self, model, val_dataset, eval_interval: int = 10):
        super(ValidationCallback, self).__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.eval_interval = eval_interval
        self.best_acc = 0.0
        self.metrics_history = []
        
    def on_train_epoch_end(self, run_context):
        """训练轮次结束时的回调"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        
        if cur_epoch % self.eval_interval == 0:
            metrics = evaluate_model(self.model, self.val_dataset)
            self.metrics_history.append(metrics)
            
            print(f"Epoch {cur_epoch:3d} - 验证指标:")
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            
            if metrics['accuracy'] > self.best_acc:
                self.best_acc = metrics['accuracy']
                print(f"  🎉 新的最佳准确率: {self.best_acc:.4f}")

class CustomLossMonitor(LossMonitor):
    """自定义损失监控器"""
    
    def __init__(self, per_print_times: int = 1):
        super(CustomLossMonitor, self).__init__(per_print_times)
        self.per_print_times = per_print_times  # 显式设置属性
        self.loss_history = []
        
    def on_train_step_end(self, run_context):
        """训练步骤结束时的回调"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
            
        self.loss_history.append(float(loss))
        
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num
        
        if cur_step_in_epoch % self.per_print_times == 0:
            print(f"Step {cur_num:5d} - Loss: {float(loss):.6f}")

def save_model(model: DiabetesNet, save_path: str, config: ModelConfig = None):
    """保存模型"""
    print(f"💾 保存模型到: {save_path}")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型参数
    ms.save_checkpoint(model, save_path)
    
    # 保存配置
    if config:
        config_path = save_path.replace('.ckpt', '_config.json')
        import json
        config_dict = {
            'input_size': config.input_size,
            'hidden_sizes': config.hidden_sizes,
            'output_size': config.output_size,
            'dropout_rate': config.dropout_rate,
            'activation': config.activation,
            'feature_means': config.feature_means,
            'feature_stds': config.feature_stds,
            'feature_names': config.feature_names
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print("✅ 模型保存完成!")

def load_model(model_path: str, config: ModelConfig = None) -> DiabetesNet:
    """加载模型"""
    print(f"📂 加载模型: {model_path}")
    
    if config is None:
        # 尝试加载配置
        config_path = model_path.replace('.ckpt', '_config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        else:
            config = ModelConfig()
    
    # 创建模型
    model = DiabetesNet(config)
    
    # 加载参数
    ms.load_checkpoint(model_path, model)
    
    print("✅ 模型加载完成!")
    return model

def evaluate_model(model: DiabetesNet, test_dataset: ds.Dataset) -> Dict[str, float]:
    """评估模型"""
    model.set_train(False)
    
    all_predictions = []
    all_targets = []
    
    for batch in test_dataset.create_dict_iterator():
        features = batch['features']
        labels = batch['label']
        
        # 预测
        predictions = model(features)
        
        all_predictions.append(predictions.asnumpy())
        all_targets.append(labels.asnumpy())
    
    # 合并所有批次
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 转换为Tensor计算指标
    pred_tensor = Tensor(all_predictions, mstype.float32)
    target_tensor = Tensor(all_targets, mstype.float32)
    
    metrics = DiabetesMetrics.compute_all_metrics(pred_tensor, target_tensor)
    
    return metrics

def predict_single_sample(model: DiabetesNet, features: np.ndarray) -> Tuple[float, float]:
    """单样本预测"""
    model.set_train(False)
    
    # 转换为Tensor
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    features_tensor = Tensor(features, mstype.float32)
    
    # 预测
    with ms.no_grad():
        prediction = model(features_tensor)
        probability = float(prediction[0, 0])
        
    # 预测标签
    prediction_label = 1 if probability > 0.5 else 0
    
    return prediction_label, probability

def predict_batch(model: DiabetesNet, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """批量预测"""
    model.set_train(False)
    
    features_tensor = Tensor(features, mstype.float32)
    
    # 预测
    with ms.no_grad():
        predictions = model(features_tensor)
        probabilities = predictions.asnumpy().flatten()
        
    # 预测标签
    prediction_labels = (probabilities > 0.5).astype(int)
    
    return prediction_labels, probabilities

def create_trainer(
    model: DiabetesNet,
    train_dataset: ds.Dataset,
    val_dataset: ds.Dataset = None,
    config: TrainingConfig = None
) -> Tuple[Model, list]:
    """创建训练器"""
    
    if config is None:
        config = TrainingConfig()
    
    # 创建损失函数
    loss_fn = DiabetesLoss()
    
    # 创建优化器
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.learning_rate)
    
    # 创建训练模型
    trainer = Model(model, loss_fn, optimizer)
    
    # 创建回调函数
    callbacks = []
    
    # 损失监控
    loss_monitor = CustomLossMonitor(per_print_times=50)
    callbacks.append(loss_monitor)
    
    # 时间监控
    time_monitor = TimeMonitor()
    callbacks.append(time_monitor)
    
    # 模型保存
    config_ckpt = CheckpointConfig(
        save_checkpoint_steps=config.save_checkpoint_steps,
        keep_checkpoint_max=config.keep_checkpoint_max
    )
    checkpoint_cb = ModelCheckpoint(
        prefix="diabetes_model",
        directory=config.model_save_path,
        config=config_ckpt
    )
    callbacks.append(checkpoint_cb)
    
    # 验证回调
    if val_dataset is not None:
        val_callback = ValidationCallback(model, val_dataset, config.eval_interval)
        callbacks.append(val_callback)
    
    return trainer, callbacks

def model_summary(model: DiabetesNet) -> Dict[str, Any]:
    """模型摘要"""
    summary = {
        'architecture': {
            'input_size': model.input_size,
            'hidden_sizes': model.hidden_sizes,
            'output_size': model.output_size,
            'dropout_rate': model.dropout_rate,
            'activation': model.config.activation
        },
        'parameters': {
            'total_params': 0,
            'trainable_params': 0
        },
        'layers': []
    }
    
    # 计算参数数量
    total_params = 0
    trainable_params = 0
    
    for param in model.get_parameters():
        param_count = param.size
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    summary['parameters']['total_params'] = total_params
    summary['parameters']['trainable_params'] = trainable_params
    
    return summary

def benchmark_model(model: DiabetesNet, input_shape: Tuple[int, ...], 
                   num_runs: int = 100) -> Dict[str, float]:
    """模型性能基准测试"""
    model.set_train(False)
    
    # 预热
    dummy_input = Tensor(np.random.randn(*input_shape), mstype.float32)
    for _ in range(10):
        _ = model(dummy_input)
    
    # 基准测试
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model(dummy_input)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    return {
        'avg_inference_time_ms': np.mean(times),
        'std_inference_time_ms': np.std(times),
        'min_inference_time_ms': np.min(times),
        'max_inference_time_ms': np.max(times),
        'throughput_samples_per_sec': input_shape[0] / (np.mean(times) / 1000)
    }

def export_model_for_serving(model: DiabetesNet, export_path: str, 
                           input_shape: Tuple[int, ...] = (1, 9)):
    """导出模型用于服务部署"""
    print(f"📤 导出模型用于服务: {export_path}")
    
    model.set_train(False)
    
    # 创建示例输入
    dummy_input = Tensor(np.random.randn(*input_shape), mstype.float32)
    
    # 导出模型
    ms.export(model, dummy_input, file_name=export_path, file_format='MINDIR')
    
    print("✅ 模型导出完成!")

if __name__ == "__main__":
    # 测试工具函数
    print("🔧 测试模型工具函数...")
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 创建测试模型
    config = ModelConfig()
    model = DiabetesNet(config)
    
    # 模型摘要
    summary = model_summary(model)
    print("模型摘要:", summary)
    
    # 性能基准测试
    benchmark = benchmark_model(model, (32, 9), num_runs=10)
    print("性能基准:", benchmark)
    
    print("✅ 工具函数测试完成!") 