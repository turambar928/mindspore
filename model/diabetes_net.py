"""
MindSpore糖尿病预测神经网络模型
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype
from mindspore.common.initializer import Normal, Uniform
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import ModelConfig

class DiabetesNet(nn.Cell):
    """糖尿病预测神经网络"""
    
    def __init__(self, config: ModelConfig = None):
        super(DiabetesNet, self).__init__()
        
        self.config = config or ModelConfig()
        
        # 网络架构
        self.input_size = self.config.input_size
        self.hidden_sizes = self.config.hidden_sizes
        self.output_size = self.config.output_size
        self.dropout_rate = self.config.dropout_rate
        
        # 构建层
        self.layers = nn.SequentialCell()
        
        # 输入层
        prev_size = self.input_size
        
        # 隐藏层
        for i, hidden_size in enumerate(self.hidden_sizes):
            # 全连接层
            self.layers.append(
                nn.Dense(
                    prev_size, 
                    hidden_size,
                    weight_init=Normal(0.02),
                    bias_init='zeros'
                )
            )
            
            # 批归一化
            self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # 激活函数
            if self.config.activation == "relu":
                self.layers.append(nn.ReLU())
            elif self.config.activation == "sigmoid":
                self.layers.append(nn.Sigmoid())
            elif self.config.activation == "tanh":
                self.layers.append(nn.Tanh())
            else:
                self.layers.append(nn.ReLU())
            
            # Dropout
            self.layers.append(nn.Dropout(p=self.dropout_rate))
            
            prev_size = hidden_size
        
        # 输出层
        self.output_layer = nn.Dense(
            prev_size, 
            self.output_size,
            weight_init=Normal(0.02),
            bias_init='zeros'
        )
        
        # 输出激活函数
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        """前向传播"""
        # 通过隐藏层
        x = self.layers(x)
        
        # 输出层
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x

class DiabetesLoss(nn.Cell):
    """糖尿病预测损失函数"""
    
    def __init__(self, pos_weight: float = 1.0):
        super(DiabetesLoss, self).__init__()
        self.pos_weight = pos_weight
        self.binary_cross_entropy = nn.BCELoss(reduction='mean')
        
    def construct(self, predictions, targets):
        """计算加权二元交叉熵损失"""
        # 重塑标签维度
        targets = targets.view(-1, 1)
        
        # 计算损失
        loss = self.binary_cross_entropy(predictions, targets)
        
        return loss

class DiabetesWithLoss(nn.Cell):
    """带损失的网络包装器"""
    
    def __init__(self, network, loss_fn):
        super(DiabetesWithLoss, self).__init__()
        self.network = network
        self.loss_fn = loss_fn
        
    def construct(self, features, labels):
        """前向传播并计算损失"""
        predictions = self.network(features)
        loss = self.loss_fn(predictions, labels)
        return loss, predictions

class DiabetesMetrics:
    """糖尿病预测评估指标"""
    
    @staticmethod
    def accuracy(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
        """计算准确率"""
        pred_labels = (predictions > threshold).astype(mstype.float32)
        targets = targets.view(-1, 1)
        correct = (pred_labels == targets).sum()
        total = targets.shape[0]
        return float(correct / total)
    
    @staticmethod
    def precision_recall_f1(predictions: Tensor, targets: Tensor, threshold: float = 0.5):
        """计算精确率、召回率和F1分数"""
        pred_labels = (predictions > threshold).astype(mstype.float32).asnumpy().flatten()
        targets = targets.asnumpy().flatten()
        
        # 计算混淆矩阵元素
        tp = np.sum((pred_labels == 1) & (targets == 1))
        fp = np.sum((pred_labels == 1) & (targets == 0))
        fn = np.sum((pred_labels == 0) & (targets == 1))
        tn = np.sum((pred_labels == 0) & (targets == 0))
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    @staticmethod
    def compute_all_metrics(predictions: Tensor, targets: Tensor, threshold: float = 0.5):
        """计算所有指标"""
        accuracy = DiabetesMetrics.accuracy(predictions, targets, threshold)
        precision, recall, f1 = DiabetesMetrics.precision_recall_f1(predictions, targets, threshold)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss: float):
        """检查是否应该早停"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

def create_model(config: ModelConfig = None) -> DiabetesNet:
    """创建模型的便捷函数"""
    if config is None:
        config = ModelConfig()
        
    model = DiabetesNet(config)
    return model

def create_loss_fn(pos_weight: float = 1.0) -> DiabetesLoss:
    """创建损失函数的便捷函数"""
    return DiabetesLoss(pos_weight)

def create_model_with_loss(config: ModelConfig = None, pos_weight: float = 1.0) -> DiabetesWithLoss:
    """创建带损失的模型"""
    model = create_model(config)
    loss_fn = create_loss_fn(pos_weight)
    return DiabetesWithLoss(model, loss_fn)

def print_model_info(model: DiabetesNet):
    """打印模型信息"""
    print("🧠 MindSpore糖尿病预测模型")
    print("=" * 50)
    print(f"输入维度: {model.input_size}")
    print(f"隐藏层尺寸: {model.hidden_sizes}")
    print(f"输出维度: {model.output_size}")
    print(f"Dropout率: {model.dropout_rate}")
    print(f"激活函数: {model.config.activation}")
    
    # 计算参数数量
    param_count = 0
    for param in model.trainable_params():
        param_count += param.size
    print(f"可训练参数数量: {param_count:,}")
    print("=" * 50)

def test_model():
    """测试模型"""
    print("🧪 测试MindSpore模型...")
    
    config = ModelConfig()
    model = create_model(config)
    
    # 打印模型信息
    print_model_info(model)
    
    # 创建测试数据
    batch_size = 16
    test_input = Tensor(np.random.randn(batch_size, config.input_size), mstype.float32)
    
    # 前向传播测试
    print(f"测试输入形状: {test_input.shape}")
    
    model.set_train(False)
    output = model(test_input)
    
    print(f"模型输出形状: {output.shape}")
    print(f"输出值范围: [{float(output.min()):.4f}, {float(output.max()):.4f}]")
    
    # 测试损失函数
    loss_fn = create_loss_fn()
    targets = Tensor(np.random.randint(0, 2, (batch_size, 1)), mstype.float32)
    loss = loss_fn(output, targets)
    print(f"测试损失: {float(loss):.4f}")
    
    # 测试指标
    metrics = DiabetesMetrics.compute_all_metrics(output, targets)
    print("测试指标:", metrics)
    
    print("✅ 模型测试完成!")

if __name__ == "__main__":
    # 设置MindSpore上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 运行测试
    test_model() 