"""
MindSpore糖尿病预测API服务
"""
import os
import json
import time
from typing import Dict, List, Any, Union
from datetime import datetime
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
import mindspore as ms
from mindspore import Tensor, dtype as mstype

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import ModelConfig, ServingConfig
from data.data_processor import DiabetesDataProcessor
from model.model_utils import load_model
from serving.inference import DiabetesPredictor

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
predictor = None
serving_config = ServingConfig()
request_count = 0
start_time = time.time()

def initialize_service():
    """初始化服务"""
    global predictor
    
    print("🚀 初始化MindSpore糖尿病预测服务...")
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 初始化预测器
    try:
        predictor = DiabetesPredictor(
            model_path=serving_config.model_path,
            config_path=None
        )
        print("✅ 服务初始化成功!")
        return True
    except Exception as e:
        print(f"❌ 服务初始化失败: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    global request_count, start_time
    
    request_count += 1
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy",
        "service": "mindspore-diabetes-prediction",
        "version": "1.0.0",
        "uptime_seconds": round(uptime, 2),
        "requests_served": request_count,
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    """单样本预测接口"""
    global request_count
    request_count += 1
    
    try:
        # 验证请求
        if not request.json:
            return jsonify({"error": "请求体必须是JSON格式"}), 400
        
        data = request.json
        
        # 验证必需字段
        required_fields = ['age', 'gender', 'bmi', 'HbA1c_level', 
                          'blood_glucose_level', 'smoking_history', 
                          'hypertension', 'heart_disease']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"缺少必需字段: {missing_fields}"
            }), 400
        
        # 进行预测
        start_time = time.time()
        result = predictor.predict_single(data)
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        response = {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "inference_time_ms": round(inference_time, 2),
            "input_data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": "预测过程中发生错误",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量预测接口"""
    global request_count
    request_count += 1
    
    try:
        # 验证请求
        if not request.json:
            return jsonify({"error": "请求体必须是JSON格式"}), 400
        
        data = request.json
        
        if 'samples' not in data:
            return jsonify({"error": "请求必须包含'samples'字段"}), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({"error": "'samples'必须是列表"}), 400
        
        if len(samples) > serving_config.max_batch_size:
            return jsonify({
                "error": f"批量大小超过限制 (最大: {serving_config.max_batch_size})"
            }), 400
        
        # 进行批量预测
        start_time = time.time()
        results = predictor.predict_batch(samples)
        inference_time = (time.time() - start_time) * 1000
        
        response = {
            "predictions": results,
            "batch_size": len(samples),
            "inference_time_ms": round(inference_time, 2),
            "avg_time_per_sample_ms": round(inference_time / len(samples), 2),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": "批量预测过程中发生错误",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """模型信息接口"""
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({
            "error": "获取模型信息失败",
            "details": str(e)
        }), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """特征重要性接口"""
    try:
        importance = predictor.get_feature_importance()
        return jsonify(importance)
    except Exception as e:
        return jsonify({
            "error": "获取特征重要性失败",
            "details": str(e)
        }), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """性能基准测试接口"""
    try:
        data = request.json
        num_samples = data.get('num_samples', 100)
        
        if num_samples > 1000:
            return jsonify({"error": "基准测试样本数量不能超过1000"}), 400
        
        result = predictor.benchmark(num_samples)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": "性能基准测试失败",
            "details": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def service_stats():
    """服务统计接口"""
    global request_count, start_time
    
    uptime = time.time() - start_time
    
    stats = {
        "service_uptime_seconds": round(uptime, 2),
        "total_requests": request_count,
        "requests_per_minute": round(request_count / (uptime / 60), 2) if uptime > 0 else 0,
        "memory_usage": predictor.get_memory_usage() if predictor else "N/A",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(stats)

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "error": "接口不存在",
        "available_endpoints": [
            "/health",
            "/predict",
            "/predict_batch", 
            "/model_info",
            "/feature_importance",
            "/benchmark",
            "/stats"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "error": "服务器内部错误",
        "details": str(error)
    }), 500

def main():
    """主函数"""
    print("🧠 MindSpore糖尿病预测API服务")
    print("=" * 50)
    
    # 初始化服务
    if not initialize_service():
        print("❌ 服务初始化失败，退出")
        return 1
    
    # 启动服务
    print(f"🌐 启动API服务...")
    print(f"地址: http://{serving_config.host}:{serving_config.port}")
    print(f"工作进程: {serving_config.workers}")
    print("=" * 50)
    
    app.run(
        host=serving_config.host,
        port=serving_config.port,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main() 