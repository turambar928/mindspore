#!/usr/bin/env python3
"""
简化的API部署脚本
"""
import os
import sys
from flask import Flask, request, jsonify
import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from model.diabetes_net import DiabetesNet
from data.data_processor import DiabetesDataProcessor
from config.model_config import ModelConfig, DataConfig

app = Flask(__name__)

# 全局变量
model = None
processor = None

def initialize_model():
    """初始化模型"""
    global model, processor
    
    print("🔄 初始化模型和数据处理器...")
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 创建配置
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # 创建数据处理器
    processor = DiabetesDataProcessor(data_config, model_config)
    
    # 创建模型
    model = DiabetesNet(model_config)
    
    # 查找模型文件
    model_paths = [
        "./checkpoints/diabetes_model.ckpt",
        "./checkpoints/best_model.ckpt",
        "./checkpoints/diabetes_model-1_2500.ckpt",
    ]
    
    model_loaded = False
    for checkpoint_path in model_paths:
        if os.path.exists(checkpoint_path):
            try:
                param_dict = load_checkpoint(checkpoint_path)
                load_param_into_net(model, param_dict)
                print(f"✅ 模型加载成功: {checkpoint_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️ 模型加载失败 {checkpoint_path}: {e}")
                continue
    
    if not model_loaded:
        print("❌ 未找到可用的模型文件，请先训练模型")
        print("可用的模型路径应该是:")
        for path in model_paths:
            print(f"  - {path}")
        return False
    
    # 设置为评估模式
    model.set_train(False)
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'message': '糖尿病预测API服务正常运行' if model_loaded else '模型未加载，请先训练模型'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """单次预测"""
    try:
        if model is None:
            return jsonify({'error': '模型未加载，请先训练模型'}), 500
        
        # 获取输入数据
        data = request.json
        
        # 验证必需字段
        required_fields = ['age', 'gender', 'bmi', 'HbA1c_level', 
                          'blood_glucose_level', 'smoking_history', 
                          'hypertension', 'heart_disease']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'缺少必需字段: {missing_fields}'}), 400
        
        # 数据类型转换
        try:
            data['age'] = float(data['age'])
            data['bmi'] = float(data['bmi'])
            data['HbA1c_level'] = float(data['HbA1c_level'])
            data['blood_glucose_level'] = float(data['blood_glucose_level'])
            data['hypertension'] = int(data['hypertension'])
            data['heart_disease'] = int(data['heart_disease'])
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'数据类型错误: {e}'}), 400
        
        # 预处理数据
        features = processor.preprocess_single_sample(data)
        
        # 转换为MindSpore张量
        features_tensor = ms.Tensor(features, ms.float32)
        
        # 预测
        output = model(features_tensor)
        probability = float(output.asnumpy()[0][0])
        
        # 转换为概率和预测结果
        prediction = 1 if probability > 0.5 else 0
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'risk_level': 'High' if prediction == 1 else 'Low',
            'message': '糖尿病风险高' if prediction == 1 else '糖尿病风险低',
            'confidence': abs(probability - 0.5) * 2  # 置信度
        })
        
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        return jsonify({'error': '模型未加载'}), 500
    
    return jsonify({
        'model_type': 'MindSpore Neural Network',
        'input_features': [
            'age', 'hypertension', 'heart_disease', 'bmi',
            'HbA1c_level', 'blood_glucose_level',
            'gender_encoded', 'smoking_history_encoded', 'age_group_encoded'
        ],
        'gender_mapping': {
            'Female': 0, 'Male': 1, 'Other': 2
        },
        'smoking_mapping': {
            'current': 0, 'not current': 1, 'ever': 2,
            'former': 3, 'never': 4, 'No Info': 5
        },
        'output': 'diabetes probability (0-1)',
        'threshold': 0.5
    })

@app.route('/test', methods=['GET'])
def test_prediction():
    """测试预测功能"""
    if model is None:
        return jsonify({'error': '模型未加载'}), 500
    
    # 测试数据
    test_data = {
        "age": 45,
        "gender": "Male",
        "bmi": 28.5,
        "HbA1c_level": 6.5,
        "blood_glucose_level": 140,
        "smoking_history": "former",
        "hypertension": 1,
        "heart_disease": 0
    }
    
    try:
        # 模拟预测请求
        features = processor.preprocess_single_sample(test_data)
        features_tensor = ms.Tensor(features, ms.float32)
        output = model(features_tensor)
        probability = float(output.asnumpy()[0][0])
        prediction = 1 if probability > 0.5 else 0
        
        return jsonify({
            'test_data': test_data,
            'prediction': prediction,
            'probability': probability,
            'status': 'success',
            'message': '测试预测成功'
        })
        
    except Exception as e:
        return jsonify({
            'test_data': test_data,
            'error': str(e),
            'status': 'failed',
            'message': '测试预测失败'
        })

def print_usage():
    """打印使用说明"""
    print("\n" + "="*60)
    print("🚀 MindSpore糖尿病预测API服务")
    print("="*60)
    print("API端点:")
    print("  GET  /health      - 健康检查")
    print("  POST /predict     - 糖尿病预测")
    print("  GET  /model_info  - 模型信息")
    print("  GET  /test        - 测试预测")
    print("\n预测示例:")
    print("curl -X POST http://localhost:5000/predict \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print('    "age": 45,')
    print('    "gender": "Male",')
    print('    "bmi": 28.5,')
    print('    "HbA1c_level": 6.5,')
    print('    "blood_glucose_level": 140,')
    print('    "smoking_history": "former",')
    print('    "hypertension": 1,')
    print('    "heart_disease": 0')
    print("  }'")
    print("\n" + "="*60)

if __name__ == '__main__':
    print("🔄 启动MindSpore糖尿病预测API...")
    
    if initialize_model():
        print_usage()
        print("\n🚀 API服务启动成功!")
        print("访问 http://localhost:5000/health 检查服务状态")
        print("访问 http://localhost:5000/test 进行测试预测")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ 模型初始化失败，API无法启动")
        print("请先运行训练脚本: python train_simple.py")
        sys.exit(1) 