"""
MindSpore糖尿病预测模型配置
"""
import os
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型配置类"""
    
    # 模型架构参数
    input_size: int = 9
    hidden_sizes: List[int] = None
    output_size: int = 1
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    # 训练参数
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # 数据预处理参数
    feature_means: List[float] = None
    feature_stds: List[float] = None
    feature_names: List[str] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 32, 16]
            
        if self.feature_means is None:
            self.feature_means = [
                41.886437, 0.07485, 0.03942, 27.31264, 5.527507, 
                138.058354, 0.41448, 2.180347, 1.663069
            ]
            
        if self.feature_stds is None:
            self.feature_stds = [
                22.517043, 0.263438, 0.194835, 6.59013, 1.070677, 
                40.708136, 0.493031, 1.889659, 1.170753
            ]
            
        if self.feature_names is None:
            self.feature_names = [
                'age', 'hypertension', 'heart_disease', 'bmi',
                'HbA1c_level', 'blood_glucose_level',
                'gender_encoded', 'smoking_history_encoded', 'age_group_encoded'
            ]

@dataclass
class DataConfig:
    """数据配置类"""
    
    # 数据路径
    data_path: str = "../diabetes_prediction_dataset.csv"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 特征编码映射
    gender_mapping: Dict[str, int] = None
    smoking_mapping: Dict[str, int] = None
    
    def __post_init__(self):
        """初始化默认映射"""
        if self.gender_mapping is None:
            self.gender_mapping = {
                'Female': 0, 'Male': 1, 'Other': 2,
                'female': 0, 'male': 1, 'other': 2
            }
            
        if self.smoking_mapping is None:
            self.smoking_mapping = {
                'current': 0, 'not current': 1, 'ever': 2,
                'former': 3, 'never': 4, 'No Info': 5,
                'no info': 5
            }

@dataclass 
class TrainingConfig:
    """训练配置类"""
    
    # 设备配置
    device_target: str = "CPU"  # "CPU", "GPU", "Ascend"
    device_id: int = 0
    
    # 保存路径
    model_save_path: str = "./checkpoints"
    log_path: str = "./logs"
    
    # 训练控制
    save_checkpoint_steps: int = 100
    keep_checkpoint_max: int = 10
    loss_scale: float = 1.0
    
    # 评估参数
    eval_interval: int = 10
    metrics: List[str] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1']
            
        # 确保目录存在
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

@dataclass
class ServingConfig:
    """服务配置类"""
    
    # API服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # 模型路径
    model_path: str = "./checkpoints/diabetes_model.ckpt"
    
    # 性能配置
    max_batch_size: int = 128
    timeout: int = 30
    
    # 华为云ModelArts配置
    modelarts_model_name: str = "mindspore-diabetes-prediction"
    modelarts_model_version: str = "1.0.0"
    modelarts_service_name: str = "diabetes-prediction-service"

# 全局配置实例
model_config = ModelConfig()
data_config = DataConfig()
training_config = TrainingConfig()
serving_config = ServingConfig()

def get_config(config_type: str = "model") -> Any:
    """获取配置实例"""
    config_map = {
        "model": model_config,
        "data": data_config, 
        "training": training_config,
        "serving": serving_config
    }
    return config_map.get(config_type, model_config)

def update_config_for_device(device: str = "CPU"):
    """根据设备类型更新配置"""
    global training_config
    
    if device.upper() == "ASCEND":
        training_config.device_target = "Ascend"
        training_config.device_id = 0
        # 昇腾NPU优化参数
        model_config.batch_size = 64
        model_config.learning_rate = 0.002
        
    elif device.upper() == "GPU":
        training_config.device_target = "GPU"
        training_config.device_id = 0
        # GPU优化参数
        model_config.batch_size = 128
        model_config.learning_rate = 0.001
        
    else:
        training_config.device_target = "CPU"
        # CPU保守参数
        model_config.batch_size = 32
        model_config.learning_rate = 0.0005

if __name__ == "__main__":
    # 配置测试
    print("模型配置:", model_config)
    print("数据配置:", data_config)
    print("训练配置:", training_config)
    print("服务配置:", serving_config) 