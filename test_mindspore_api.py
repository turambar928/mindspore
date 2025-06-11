"""
测试MindSpore API服务脚本
"""
import requests
import json
import time
from typing import Dict, Any

# API基础URL
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("🔍 测试健康检查...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 健康检查通过")
            print(f"  服务状态: {result.get('status')}")
            print(f"  运行时间: {result.get('uptime_seconds')}秒")
            print(f"  已处理请求: {result.get('requests_served')}")
            return True
        else:
            print(f"❌ 健康检查失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 健康检查异常: {str(e)}")
        return False

def test_single_prediction():
    """测试单样本预测"""
    print("\n🧪 测试单样本预测...")
    
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
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict", 
            json=test_data, 
            timeout=10
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 单样本预测成功")
            print(f"  预测结果: {result.get('prediction_text')}")
            print(f"  概率: {result.get('probability'):.3f}")
            print(f"  置信度: {result.get('confidence')}")
            print(f"  风险等级: {result.get('risk_level')}")
            print(f"  响应时间: {(end_time - start_time)*1000:.2f}ms")
            return True
        else:
            print(f"❌ 单样本预测失败: HTTP {response.status_code}")
            print(f"  错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 单样本预测异常: {str(e)}")
        return False

def test_batch_prediction():
    """测试批量预测"""
    print("\n🧪 测试批量预测...")
    
    # 批量测试数据
    test_samples = [
        {
            "age": 45, "gender": "Male", "bmi": 28.5,
            "HbA1c_level": 6.5, "blood_glucose_level": 140,
            "smoking_history": "former", "hypertension": 1, "heart_disease": 0
        },
        {
            "age": 30, "gender": "Female", "bmi": 22.0,
            "HbA1c_level": 5.0, "blood_glucose_level": 90,
            "smoking_history": "never", "hypertension": 0, "heart_disease": 0
        },
        {
            "age": 65, "gender": "Male", "bmi": 35.0,
            "HbA1c_level": 8.5, "blood_glucose_level": 200,
            "smoking_history": "current", "hypertension": 1, "heart_disease": 1
        }
    ]
    
    batch_data = {"samples": test_samples}
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict_batch", 
            json=batch_data, 
            timeout=15
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 批量预测成功")
            print(f"  批次大小: {result.get('batch_size')}")
            print(f"  总响应时间: {(end_time - start_time)*1000:.2f}ms")
            print(f"  平均每样本: {result.get('avg_time_per_sample_ms'):.2f}ms")
            
            # 显示前几个预测结果
            predictions = result.get('predictions', [])
            for i, pred in enumerate(predictions[:3]):
                print(f"  样本{i+1}: {pred.get('prediction_text')} (概率: {pred.get('probability'):.3f})")
            
            return True
        else:
            print(f"❌ 批量预测失败: HTTP {response.status_code}")
            print(f"  错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 批量预测异常: {str(e)}")
        return False

def test_model_info():
    """测试模型信息接口"""
    print("\n🧪 测试模型信息...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model_info", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 模型信息获取成功")
            
            arch = result.get('model_architecture', {})
            print(f"  输入维度: {arch.get('input_size')}")
            print(f"  隐藏层: {arch.get('hidden_sizes')}")
            print(f"  激活函数: {arch.get('activation')}")
            
            params = result.get('parameters', {})
            print(f"  参数数量: {params.get('total_trainable_params'):,}")
            print(f"  模型大小: {params.get('model_size_mb'):.2f}MB")
            
            return True
        else:
            print(f"❌ 模型信息获取失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 模型信息获取异常: {str(e)}")
        return False

def test_feature_importance():
    """测试特征重要性接口"""
    print("\n🧪 测试特征重要性...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/feature_importance", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 特征重要性获取成功")
            
            importance = result.get('feature_importance', {})
            top_features = result.get('top_features', [])
            
            print("  前5个重要特征:")
            for i, feature in enumerate(top_features[:5]):
                print(f"    {i+1}. {feature}: {importance.get(feature, 0):.3f}")
            
            return True
        else:
            print(f"❌ 特征重要性获取失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 特征重要性获取异常: {str(e)}")
        return False

def test_benchmark():
    """测试性能基准"""
    print("\n🧪 测试性能基准...")
    
    try:
        benchmark_data = {"num_samples": 50}
        response = requests.post(
            f"{API_BASE_URL}/benchmark", 
            json=benchmark_data, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 性能基准测试成功")
            
            single = result.get('single_prediction', {})
            batch = result.get('batch_prediction', {})
            
            print(f"  单样本平均时间: {single.get('avg_time_ms'):.2f}ms")
            print(f"  批量吞吐量: {batch.get('throughput_samples_per_sec'):.1f} 样本/秒")
            
            return True
        else:
            print(f"❌ 性能基准测试失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 性能基准测试异常: {str(e)}")
        return False

def test_service_stats():
    """测试服务统计"""
    print("\n🧪 测试服务统计...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 服务统计获取成功")
            print(f"  运行时间: {result.get('service_uptime_seconds')}秒")
            print(f"  总请求数: {result.get('total_requests')}")
            print(f"  每分钟请求: {result.get('requests_per_minute'):.1f}")
            return True
        else:
            print(f"❌ 服务统计获取失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 服务统计获取异常: {str(e)}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🧠 MindSpore API全面测试")
    print("=" * 50)
    
    tests = [
        ("健康检查", test_health_check),
        ("单样本预测", test_single_prediction),
        ("批量预测", test_batch_prediction),
        ("模型信息", test_model_info),
        ("特征重要性", test_feature_importance),
        ("性能基准", test_benchmark),
        ("服务统计", test_service_stats)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            time.sleep(0.5)  # 短暂延迟
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 异常: {str(e)}")
    
    print(f"\n📊 测试结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过! MindSpore API运行正常")
    else:
        print("⚠️ 部分测试失败，请检查API服务状态")
    
    return passed_tests == total_tests

def main():
    """主函数"""
    print("🧪 MindSpore API测试工具")
    print(f"API地址: {API_BASE_URL}")
    print("=" * 50)
    
    # 首先检查服务是否可达
    print("🔍 检查API服务可达性...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API服务可达")
        else:
            print(f"❌ API服务响应异常: HTTP {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 无法连接到API服务: {str(e)}")
        print("请确保MindSpore API服务正在运行")
        return
    
    # 运行所有测试
    run_all_tests()

if __name__ == "__main__":
    main() 