"""
MindSpore糖尿病预测数据处理模块
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor, dtype as mstype

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import DataConfig, ModelConfig

class DiabetesDataProcessor:
    """糖尿病数据处理器"""
    
    def __init__(self, data_config: DataConfig = None, model_config: ModelConfig = None):
        self.data_config = data_config or DataConfig()
        self.model_config = model_config or ModelConfig()
        self.scaler = StandardScaler()
        self.feature_stats = {
            'means': np.array(self.model_config.feature_means),
            'stds': np.array(self.model_config.feature_stds)
        }
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """加载数据"""
        if data_path is None:
            data_path = self.data_config.data_path
            
        print(f"📊 加载数据: {data_path}")
        data = pd.read_csv(data_path)
        data = data.dropna()
        
        print(f"数据形状: {data.shape}")
        print(f"糖尿病阳性比例: {data['diabetes'].mean():.2%}")
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        data = data.copy()
        
        # 数据清理
        data['age'] = data['age'].clip(upper=100)
        data['bmi'] = data['bmi'].clip(10, 60)
        
        # 性别编码
        data['gender_encoded'] = data['gender'].map(self.data_config.gender_mapping).fillna(0)
        
        # 吸烟史编码
        data['smoking_history_encoded'] = data['smoking_history'].map(self.data_config.smoking_mapping).fillna(4)
        
        # 年龄组编码
        def encode_age_group(age):
            if age <= 30:
                return 3  # young
            elif age <= 45:
                return 1  # middle_aged
            elif age <= 60:
                return 2  # senior
            else:
                return 0  # elderly
                
        data['age_group_encoded'] = data['age'].apply(encode_age_group)
        
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预处理数据"""
        print("🔄 开始数据预处理...")
        
        # 编码分类特征
        data = self.encode_categorical_features(data)
        
        # 选择特征
        X = data[self.model_config.feature_names].values.astype(np.float32)
        y = data['diabetes'].values.astype(np.float32)
        
        # 标准化特征
        X = self.normalize_features(X)
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"标签形状: {y.shape}")
        print(f"正样本比例: {y.mean():.2%}")
        
        return X, y
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """标准化特征"""
        means = self.feature_stats['means']
        stds = self.feature_stats['stds']
        
        X_normalized = (X - means) / stds
        return X_normalized.astype(np.float32)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """拆分数据集"""
        print("✂️ 拆分数据集...")
        
        # 先分出训练+验证集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.data_config.test_ratio,
            random_state=random_state,
            stratify=y
        )
        
        # 再分出训练集和验证集
        val_size = self.data_config.val_ratio / (1 - self.data_config.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本") 
        print(f"测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataset(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int = 32, shuffle: bool = True) -> ds.Dataset:
        """创建MindSpore数据集"""
        
        def data_generator():
            """数据生成器"""
            for i in range(len(X)):
                yield X[i], y[i]
        
        # 创建数据集
        dataset = ds.GeneratorDataset(
            source=data_generator,
            column_names=['features', 'label'],
            shuffle=shuffle
        )
        
        # 设置数据类型
        type_cast_op = ds.transforms.TypeCast(mstype.float32)
        dataset = dataset.map(operations=type_cast_op, input_columns="features")
        dataset = dataset.map(operations=type_cast_op, input_columns="label")
        
        # 批处理
        dataset = dataset.batch(batch_size)
        
        return dataset
    
    def preprocess_single_sample(self, sample_data: Dict[str, Any]) -> np.ndarray:
        """预处理单个样本（用于API推理）"""
        # 编码分类特征
        processed_data = sample_data.copy()
        
        # 性别编码
        processed_data['gender_encoded'] = self.data_config.gender_mapping.get(
            processed_data.get('gender', 'Female'), 0
        )
        
        # 吸烟史编码  
        processed_data['smoking_history_encoded'] = self.data_config.smoking_mapping.get(
            processed_data.get('smoking_history', 'never'), 4
        )
        
        # 年龄组编码
        age = processed_data.get('age', 0)
        if age <= 30:
            age_group_encoded = 3
        elif age <= 45:
            age_group_encoded = 1
        elif age <= 60:
            age_group_encoded = 2
        else:
            age_group_encoded = 0
        processed_data['age_group_encoded'] = age_group_encoded
        
        # 提取特征
        features = []
        for feature_name in self.model_config.feature_names:
            features.append(processed_data.get(feature_name, 0))
        
        features = np.array(features, dtype=np.float32)
        
        # 标准化
        features = self.normalize_features(features.reshape(1, -1))
        
        return features
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            'total_samples': len(data),
            'feature_count': len(self.model_config.feature_names),
            'positive_ratio': data['diabetes'].mean(),
            'missing_values': data.isnull().sum().sum(),
            'feature_stats': {}
        }
        
        # 数值特征统计
        numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        for feature in numeric_features:
            if feature in data.columns:
                stats['feature_stats'][feature] = {
                    'mean': data[feature].mean(),
                    'std': data[feature].std(),
                    'min': data[feature].min(),
                    'max': data[feature].max()
                }
        
        # 分类特征统计
        categorical_features = ['gender', 'smoking_history']
        for feature in categorical_features:
            if feature in data.columns:
                stats['feature_stats'][feature] = data[feature].value_counts().to_dict()
        
        return stats

def prepare_data_for_training(data_path: str = None, 
                            batch_size: int = 32) -> Tuple[ds.Dataset, ds.Dataset, ds.Dataset]:
    """准备训练数据的便捷函数"""
    
    processor = DiabetesDataProcessor()
    
    # 加载和预处理数据
    data = processor.load_data(data_path)
    X, y = processor.preprocess_data(data)
    
    # 拆分数据
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # 创建数据集
    train_dataset = processor.create_dataset(X_train, y_train, batch_size, shuffle=True)
    val_dataset = processor.create_dataset(X_val, y_val, batch_size, shuffle=False)
    test_dataset = processor.create_dataset(X_test, y_test, batch_size, shuffle=False)
    
    print("✅ 数据准备完成!")
    
    return train_dataset, val_dataset, test_dataset

def create_sample_data() -> Dict[str, Any]:
    """创建样本数据用于测试"""
    return {
        "age": 45,
        "gender": "Male",
        "bmi": 28.5,
        "HbA1c_level": 6.5,
        "blood_glucose_level": 140,
        "smoking_history": "former",
        "hypertension": 1,
        "heart_disease": 0
    }

if __name__ == "__main__":
    # 测试数据处理
    processor = DiabetesDataProcessor()
    
    # 测试样本预处理
    sample = create_sample_data()
    processed_sample = processor.preprocess_single_sample(sample)
    print("处理后的样本:", processed_sample.shape)
    
    # 如果有数据文件，测试完整流程
    data_path = "../diabetes_prediction_dataset.csv"
    if os.path.exists(data_path):
        print("\n测试完整数据处理流程...")
        train_ds, val_ds, test_ds = prepare_data_for_training(data_path, batch_size=32)
        print(f"训练数据集大小: {train_ds.get_dataset_size()}")
        print(f"验证数据集大小: {val_ds.get_dataset_size()}")
        print(f"测试数据集大小: {test_ds.get_dataset_size()}") 