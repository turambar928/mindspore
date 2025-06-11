"""
MindSpore糖尿病预测推理逻辑
"""
import os
import time
import json
import psutil
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import mindspore as ms
from mindspore import Tensor, dtype as mstype

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import ModelConfig
from data.data_processor import DiabetesDataProcessor
from model.model_utils import load_model, predict_single_sample

class DiabetesPredictor:
    """糖尿病预测器"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径（可选）
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.data_processor = None
        self.is_loaded = False
        
        # 性能统计
        self.prediction_count = 0
        self.total_inference_time = 0.0
        
        # 加载模型和配置
        self._load_model_and_config()
        
    def _load_model_and_config(self):
        """加载模型和配置"""
        try:
            print(f"📂 加载模型: {self.model_path}")
            
            # 加载配置
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                self.config = ModelConfig(**config_dict)
            else:
                # 尝试自动查找配置文件
                auto_config_path = self.model_path.replace('.ckpt', '_config.json')
                if os.path.exists(auto_config_path):
                    with open(auto_config_path, 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                    self.config = ModelConfig(**config_dict)
                else:
                    self.config = ModelConfig()
            
            # 加载模型
            self.model = load_model(self.model_path, self.config)
            
            # 初始化数据处理器
            self.data_processor = DiabetesDataProcessor(model_config=self.config)
            
            self.is_loaded = True
            print("✅ 模型加载成功!")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        单样本预测
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            预测结果字典
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        try:
            # 预处理输入数据
            features = self.data_processor.preprocess_single_sample(input_data)
            
            # 进行预测
            prediction_label, probability = predict_single_sample(self.model, features)
            
            # 计算置信度和风险等级
            confidence = self._calculate_confidence(probability)
            risk_level = self._get_risk_level(probability)
            
            # 更新统计
            inference_time = time.time() - start_time
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
            return {
                "prediction": int(prediction_label),
                "probability": float(probability),
                "confidence": confidence,
                "risk_level": risk_level,
                "prediction_text": "有糖尿病" if prediction_label == 1 else "无糖尿病",
                "inference_time_ms": round(inference_time * 1000, 2)
            }
            
        except Exception as e:
            raise RuntimeError(f"预测失败: {str(e)}")
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            input_data_list: 输入数据列表
            
        Returns:
            预测结果列表
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        try:
            # 预处理所有输入数据
            features_list = []
            for input_data in input_data_list:
                features = self.data_processor.preprocess_single_sample(input_data)
                features_list.append(features.flatten())
            
            # 批量预测
            batch_features = np.array(features_list)
            from model.model_utils import predict_batch
            prediction_labels, probabilities = predict_batch(self.model, batch_features)
            
            # 构建结果
            results = []
            for i, (label, prob) in enumerate(zip(prediction_labels, probabilities)):
                confidence = self._calculate_confidence(prob)
                risk_level = self._get_risk_level(prob)
                
                results.append({
                    "sample_id": i,
                    "prediction": int(label),
                    "probability": float(prob),
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "prediction_text": "有糖尿病" if label == 1 else "无糖尿病"
                })
            
            # 更新统计
            inference_time = time.time() - start_time
            self.prediction_count += len(input_data_list)
            self.total_inference_time += inference_time
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"批量预测失败: {str(e)}")
    
    def _calculate_confidence(self, probability: float) -> str:
        """计算置信度等级"""
        if probability >= 0.9 or probability <= 0.1:
            return "高"
        elif probability >= 0.75 or probability <= 0.25:
            return "中"
        else:
            return "低"
    
    def _get_risk_level(self, probability: float) -> str:
        """获取风险等级"""
        if probability >= 0.8:
            return "高风险"
        elif probability >= 0.6:
            return "中高风险"
        elif probability >= 0.4:
            return "中等风险"
        elif probability >= 0.2:
            return "低风险"
        else:
            return "极低风险"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_loaded:
            return {"error": "模型未加载"}
        
        # 计算模型参数数量
        total_params = sum(p.size for p in self.model.trainable_params())
        
        info = {
            "model_path": self.model_path,
            "model_architecture": {
                "input_size": self.config.input_size,
                "hidden_sizes": self.config.hidden_sizes,
                "output_size": self.config.output_size,
                "dropout_rate": self.config.dropout_rate,
                "activation": self.config.activation
            },
            "parameters": {
                "total_trainable_params": int(total_params),
                "model_size_mb": os.path.getsize(self.model_path) / (1024 * 1024)
            },
            "features": {
                "feature_names": self.config.feature_names,
                "feature_count": len(self.config.feature_names)
            },
            "statistics": {
                "predictions_made": self.prediction_count,
                "avg_inference_time_ms": round(
                    (self.total_inference_time / self.prediction_count * 1000) 
                    if self.prediction_count > 0 else 0, 2
                )
            },
            "status": "loaded" if self.is_loaded else "not_loaded"
        }
        
        return info
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """获取特征重要性"""
        feature_importance = {
            "HbA1c_level": 0.25,
            "blood_glucose_level": 0.22,
            "bmi": 0.18,
            "age": 0.15,
            "smoking_history_encoded": 0.08,
            "hypertension": 0.06,
            "heart_disease": 0.04,
            "gender_encoded": 0.02,
            "age_group_encoded": 0.01
        }
        
        # 按重要性排序
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return {
            "feature_importance": dict(sorted_features),
            "top_features": [f[0] for f in sorted_features[:5]],
            "note": "这是基于领域知识的近似重要性"
        }
    
    def benchmark(self, num_samples: int = 100) -> Dict[str, Any]:
        """性能基准测试"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        print(f"🏃 开始性能基准测试 ({num_samples} 样本)...")
        
        # 生成随机测试数据
        test_data = []
        for _ in range(num_samples):
            sample = {
                "age": np.random.randint(20, 80),
                "gender": np.random.choice(["Male", "Female"]),
                "bmi": np.random.uniform(18, 40),
                "HbA1c_level": np.random.uniform(4, 10),
                "blood_glucose_level": np.random.uniform(80, 300),
                "smoking_history": np.random.choice(["never", "former", "current"]),
                "hypertension": np.random.choice([0, 1]),
                "heart_disease": np.random.choice([0, 1])
            }
            test_data.append(sample)
        
        # 单样本基准测试
        single_times = []
        for sample in test_data[:min(50, num_samples)]:
            start_time = time.time()
            _ = self.predict_single(sample)
            single_times.append((time.time() - start_time) * 1000)
        
        # 批量基准测试
        start_time = time.time()
        _ = self.predict_batch(test_data)
        batch_time = (time.time() - start_time) * 1000
        
        return {
            "single_prediction": {
                "avg_time_ms": round(np.mean(single_times), 2),
                "min_time_ms": round(np.min(single_times), 2),
                "max_time_ms": round(np.max(single_times), 2),
                "std_time_ms": round(np.std(single_times), 2),
                "samples_tested": len(single_times)
            },
            "batch_prediction": {
                "total_time_ms": round(batch_time, 2),
                "avg_time_per_sample_ms": round(batch_time / num_samples, 2),
                "throughput_samples_per_sec": round(num_samples / (batch_time / 1000), 2),
                "batch_size": num_samples
            }
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
            "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
            "percent": round(process.memory_percent(), 2)
        }

def create_sample_input() -> Dict[str, Any]:
    """创建示例输入数据"""
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
    # 测试推理器
    print("🧪 测试MindSpore推理器...")
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 注意：这里需要实际的模型文件
    model_path = "./checkpoints/diabetes_model.ckpt"
    
    if os.path.exists(model_path):
        try:
            predictor = DiabetesPredictor(model_path)
            
            # 测试单样本预测
            sample = create_sample_input()
            result = predictor.predict_single(sample)
            print("单样本预测结果:", result)
            
            # 测试批量预测
            batch_samples = [sample, sample, sample]
            batch_results = predictor.predict_batch(batch_samples)
            print("批量预测结果:", len(batch_results))
            
            # 获取模型信息
            model_info = predictor.get_model_info()
            print("模型信息:", model_info)
            
            print("✅ 推理器测试完成!")
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
    else:
        print(f"⚠️ 模型文件不存在: {model_path}")
        print("请先训练模型") 