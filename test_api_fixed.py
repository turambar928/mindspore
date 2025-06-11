#!/usr/bin/env python3
"""
API测试脚本
"""
import requests
import json
import time

# API地址
API_URL = "http://localhost:5000"

def test_health():
    """测试健康检查"""
    print("🔍 测试健康检查...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response.status_code == 200 and result.get('model_loaded', False)
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_model_info():
    """测试模型信息"""
    print("\n🔍 测试模型信息...")
    try:
        response = requests.get(f"{API_URL}/model_info", timeout=10)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 模型信息获取失败: {e}")
        return False

def test_prediction():
    """测试预测功能"""
    print("\n🔍 测试预测功能...")
    
    # 测试数据 - 高风险患者
    high_risk_data = {
        "age": 65,
        "gender": "Male",
        "bmi": 32.5,
        "HbA1c_level": 7.2,
        "blood_glucose_level": 180,
        "smoking_history": "current",
        "hypertension": 1,
        "heart_disease": 1
    }
    
    # 测试数据 - 低风险患者
    low_risk_data = {
        "age": 25,
        "gender": "Female",
        "bmi": 22.0,
        "HbA1c_level": 5.0,
        "blood_glucose_level": 90,
        "smoking_history": "never",
        "hypertension": 0,
        "heart_disease": 0
    }
    
    test_cases = [
        ("高风险患者", high_risk_data),
        ("低风险患者", low_risk_data)
    ]
    
    all_passed = True
    
    for case_name, test_data in test_cases:
        print(f"\n  测试案例: {case_name}")
        print(f"  输入数据: {json.dumps(test_data, indent=4, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            print(f"  状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"  预测结果: {result['prediction']} ({'糖尿病' if result['prediction'] == 1 else '正常'})")
                print(f"  概率: {result['probability']:.4f}")
                print(f"  风险等级: {result['risk_level']}")
                print(f"  置信度: {result.get('confidence', 0):.4f}")
                print(f"  说明: {result['message']}")
            else:
                print(f"  ❌ 预测失败: {response.text}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ 请求失败: {e}")
            all_passed = False
    
    return all_passed

def test_internal_test_endpoint():
    """测试内置测试端点"""
    print("\n🔍 测试内置测试端点...")
    try:
        response = requests.get(f"{API_URL}/test", timeout=10)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response.status_code == 200 and result.get('status') == 'success'
    except Exception as e:
        print(f"❌ 内置测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n🔍 测试错误处理...")
    
    # 测试缺少字段
    incomplete_data = {
        "age": 45,
        "gender": "Male"
        # 缺少其他必需字段
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=incomplete_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"  状态码: {response.status_code}")
        
        if response.status_code == 400:
            result = response.json()
            print(f"  错误信息: {result.get('error', '未知错误')}")
            print("  ✅ 错误处理正常")
            return True
        else:
            print("  ❌ 错误处理异常")
            return False
            
    except Exception as e:
        print(f"  ❌ 错误处理测试失败: {e}")
        return False

def wait_for_api():
    """等待API启动"""
    print("⏳ 等待API服务启动...")
    
    for i in range(30):  # 最多等待30秒
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API服务已启动")
                return True
        except:
            pass
        
        print(f"  等待中... ({i+1}/30)")
        time.sleep(1)
    
    print("❌ API服务启动超时")
    return False

def main():
    """主测试函数"""
    print("🧪 开始API测试...")
    print("="*60)
    
    # 等待API启动
    if not wait_for_api():
        print("❌ API服务未启动，测试终止")
        return False
    
    # 执行测试
    tests = [
        ("健康检查", test_health),
        ("模型信息", test_model_info),
        ("内置测试", test_internal_test_endpoint),
        ("预测功能", test_prediction),
        ("错误处理", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 执行测试: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    # 测试结果
    print(f"\n{'='*60}")
    print(f"🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! API服务正常运行")
        return True
    else:
        print(f"⚠️ {total - passed} 个测试失败，请检查API服务")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 