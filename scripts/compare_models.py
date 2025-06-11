"""
XGBoost vs MindSpore模型对比脚本
"""
import os
import time
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, List
import matplotlib.pyplot as plt

def test_xgboost_api(api_url: str, sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """测试XGBoost API"""
    try:
        response = requests.post(f"{api_url}/predict", json=sample_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API错误: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def test_mindspore_api(api_url: str, sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """测试MindSpore API"""
    try:
        response = requests.post(f"{api_url}/predict", json=sample_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API错误: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def benchmark_models(xgboost_url: str, mindspore_url: str, num_samples: int = 100):
    """基准测试两个模型"""
    print(f"🏃 开始模型基准测试 ({num_samples} 样本)")
    
    # 生成测试数据
    test_samples = []
    for i in range(num_samples):
        sample = {
            "age": np.random.randint(20, 80),
            "gender": np.random.choice(["Male", "Female"]),
            "bmi": round(np.random.uniform(18, 40), 1),
            "HbA1c_level": round(np.random.uniform(4, 10), 1),
            "blood_glucose_level": np.random.randint(80, 300),
            "smoking_history": np.random.choice(["never", "former", "current"]),
            "hypertension": np.random.choice([0, 1]),
            "heart_disease": np.random.choice([0, 1])
        }
        test_samples.append(sample)
    
    # XGBoost测试
    print("🧪 测试XGBoost模型...")
    xgboost_times = []
    xgboost_predictions = []
    
    for sample in test_samples:
        start_time = time.time()
        result = test_xgboost_api(xgboost_url, sample)
        end_time = time.time()
        
        if "error" not in result:
            xgboost_times.append((end_time - start_time) * 1000)
            xgboost_predictions.append(result.get("prediction", 0))
        else:
            print(f"XGBoost API错误: {result['error']}")
    
    # MindSpore测试  
    print("🧪 测试MindSpore模型...")
    mindspore_times = []
    mindspore_predictions = []
    
    for sample in test_samples:
        start_time = time.time()
        result = test_mindspore_api(mindspore_url, sample)
        end_time = time.time()
        
        if "error" not in result:
            mindspore_times.append((end_time - start_time) * 1000)
            mindspore_predictions.append(result.get("prediction", 0))
        else:
            print(f"MindSpore API错误: {result['error']}")
    
    # 计算统计
    xgboost_stats = {
        "avg_time_ms": np.mean(xgboost_times) if xgboost_times else 0,
        "min_time_ms": np.min(xgboost_times) if xgboost_times else 0,
        "max_time_ms": np.max(xgboost_times) if xgboost_times else 0,
        "std_time_ms": np.std(xgboost_times) if xgboost_times else 0,
        "successful_predictions": len(xgboost_times)
    }
    
    mindspore_stats = {
        "avg_time_ms": np.mean(mindspore_times) if mindspore_times else 0,
        "min_time_ms": np.min(mindspore_times) if mindspore_times else 0,
        "max_time_ms": np.max(mindspore_times) if mindspore_times else 0,
        "std_time_ms": np.std(mindspore_times) if mindspore_times else 0,
        "successful_predictions": len(mindspore_times)
    }
    
    # 预测一致性分析
    if len(xgboost_predictions) > 0 and len(mindspore_predictions) > 0:
        min_len = min(len(xgboost_predictions), len(mindspore_predictions))
        agreement = np.mean([
            xgboost_predictions[i] == mindspore_predictions[i] 
            for i in range(min_len)
        ])
    else:
        agreement = 0
    
    comparison_results = {
        "xgboost": xgboost_stats,
        "mindspore": mindspore_stats,
        "prediction_agreement": agreement,
        "speed_improvement": (
            xgboost_stats["avg_time_ms"] / mindspore_stats["avg_time_ms"]
            if mindspore_stats["avg_time_ms"] > 0 else 0
        )
    }
    
    return comparison_results

def generate_comparison_report(results: Dict[str, Any]):
    """生成对比报告"""
    print("\n📊 模型对比报告")
    print("=" * 60)
    
    print("XGBoost模型:")
    print(f"  平均响应时间: {results['xgboost']['avg_time_ms']:.2f}ms")
    print(f"  成功预测数: {results['xgboost']['successful_predictions']}")
    
    print("\nMindSpore模型:")
    print(f"  平均响应时间: {results['mindspore']['avg_time_ms']:.2f}ms")
    print(f"  成功预测数: {results['mindspore']['successful_predictions']}")
    
    print(f"\n预测一致性: {results['prediction_agreement']:.2%}")
    
    if results['speed_improvement'] > 0:
        print(f"MindSpore速度提升: {results['speed_improvement']:.1f}x")
    
    # 保存结果
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 对比报告已保存到 model_comparison_results.json")

def plot_performance_comparison(results: Dict[str, Any]):
    """绘制性能对比图"""
    models = ['XGBoost', 'MindSpore']
    avg_times = [
        results['xgboost']['avg_time_ms'],
        results['mindspore']['avg_time_ms']
    ]
    
    plt.figure(figsize=(10, 6))
    
    # 响应时间对比
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, avg_times, color=['#FF6B6B', '#4ECDC4'])
    plt.title('平均响应时间对比')
    plt.ylabel('时间 (ms)')
    
    # 在柱子上显示数值
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time_val:.1f}ms', ha='center', va='bottom')
    
    # 成功预测数对比
    plt.subplot(1, 2, 2)
    success_counts = [
        results['xgboost']['successful_predictions'],
        results['mindspore']['successful_predictions']
    ]
    bars = plt.bar(models, success_counts, color=['#FF6B6B', '#4ECDC4'])
    plt.title('成功预测数对比')
    plt.ylabel('预测数量')
    
    # 在柱子上显示数值
    for bar, count in zip(bars, success_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🧠 XGBoost vs MindSpore模型对比工具")
    print("=" * 50)
    
    # API地址配置
    xgboost_url = "http://1.94.9.72:5000"  # 原XGBoost API
    mindspore_url = "http://localhost:8000"  # MindSpore API
    
    print(f"XGBoost API: {xgboost_url}")
    print(f"MindSpore API: {mindspore_url}")
    
    try:
        # 健康检查
        print("\n🔍 检查API健康状态...")
        
        xgb_health = test_xgboost_api(xgboost_url, {})
        ms_health = test_mindspore_api(mindspore_url, {})
        
        print(f"XGBoost API状态: {'✅' if 'error' not in str(xgb_health) else '❌'}")
        print(f"MindSpore API状态: {'✅' if 'error' not in str(ms_health) else '❌'}")
        
        # 进行基准测试
        results = benchmark_models(xgboost_url, mindspore_url, num_samples=50)
        
        # 生成报告
        generate_comparison_report(results)
        
        # 绘制对比图
        plot_performance_comparison(results)
        
        print("\n🎉 模型对比完成!")
        
    except Exception as e:
        print(f"❌ 对比过程发生错误: {str(e)}")

if __name__ == "__main__":
    main() 