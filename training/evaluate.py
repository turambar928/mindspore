"""
MindSpore糖尿病预测模型评估脚本
"""
import os
import argparse
import mindspore as ms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import ModelConfig
from data.data_processor import prepare_data_for_training, DiabetesDataProcessor
from model.model_utils import load_model, evaluate_model

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MindSpore糖尿病预测模型评估')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--data_path', type=str, 
                       default="../diabetes_prediction_dataset.csv",
                       help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--device', type=str, default='CPU',
                       choices=['CPU', 'GPU', 'Ascend'],
                       help='计算设备')
    
    return parser.parse_args()

def detailed_evaluation(model, test_dataset, output_dir):
    """详细评估模型"""
    print("🔍 进行详细模型评估...")
    
    model.set_train(False)
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    # 收集所有预测结果
    for batch in test_dataset.create_dict_iterator():
        features = batch['features']
        labels = batch['label']
        
        predictions = model(features)
        probabilities = predictions.asnumpy().flatten()
        targets = labels.asnumpy().flatten()
        
        all_probabilities.extend(probabilities)
        all_targets.extend(targets)
        all_predictions.extend((probabilities > 0.5).astype(int))
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # 基本指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision': precision_score(all_targets, all_predictions),
        'recall': recall_score(all_targets, all_predictions),
        'f1': f1_score(all_targets, all_predictions)
    }
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
    roc_auc = auc(fpr, tpr)
    
    # 保存结果
    save_evaluation_results(metrics, cm, fpr, tpr, roc_auc, output_dir)
    
    # 绘制图表
    plot_evaluation_charts(metrics, cm, fpr, tpr, roc_auc, output_dir)
    
    return metrics, cm, roc_auc

def save_evaluation_results(metrics, cm, fpr, tpr, roc_auc, output_dir):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存文本报告
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("MindSpore糖尿病预测模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("性能指标:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\nROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("混淆矩阵:\n")
        f.write(f"                预测\n")
        f.write(f"实际     0    1\n")
        f.write(f"  0   {cm[0,0]:4d} {cm[0,1]:4d}\n")
        f.write(f"  1   {cm[1,0]:4d} {cm[1,1]:4d}\n")

def plot_evaluation_charts(metrics, cm, fpr, tpr, roc_auc, output_dir):
    """绘制评估图表"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 指标条形图
    ax1 = axes[0, 0]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    bars = ax1.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax1.set_title('模型性能指标')
    ax1.set_ylabel('分数')
    ax1.set_ylim(0, 1)
    
    # 在条形图上添加数值
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 混淆矩阵热力图
    ax2 = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['预测 0', '预测 1'],
                yticklabels=['实际 0', '实际 1'])
    ax2.set_title('混淆矩阵')
    
    # ROC曲线
    ax3 = axes[1, 0]
    ax3.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('假阳性率')
    ax3.set_ylabel('真阳性率')
    ax3.set_title('ROC曲线')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)
    
    # 概率分布
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.5, f'模型评估完成\n\n准确率: {metrics["accuracy"]:.3f}\n'
                       f'精确率: {metrics["precision"]:.3f}\n'
                       f'召回率: {metrics["recall"]:.3f}\n'
                       f'F1分数: {metrics["f1"]:.3f}\n'
                       f'ROC AUC: {roc_auc:.3f}',
             transform=ax4.transAxes, fontsize=12,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_title('评估摘要')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_charts.png'), dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_baseline(metrics, output_dir):
    """与基线模型比较"""
    # 简单基线：始终预测多数类
    baseline_accuracy = 0.913  # 来自数据分析
    
    improvement = {
        'accuracy_improvement': metrics['accuracy'] - baseline_accuracy,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }
    
    print(f"\n📊 与基线模型比较:")
    print(f"基线准确率 (多数类): {baseline_accuracy:.3f}")
    print(f"模型准确率: {metrics['accuracy']:.3f}")
    print(f"准确率提升: {improvement['accuracy_improvement']:.3f}")
    
    return improvement

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置环境
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device)
    
    # 检查文件存在性
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型文件不存在: {args.model_path}")
        return 1
    
    if not os.path.exists(args.data_path):
        print(f"❌ 错误: 数据文件不存在: {args.data_path}")
        return 1
    
    print("🧠 MindSpore糖尿病预测模型评估")
    print("=" * 50)
    
    try:
        # 加载模型
        print("📂 加载模型...")
        model = load_model(args.model_path)
        
        # 准备数据
        print("📊 准备测试数据...")
        _, _, test_dataset = prepare_data_for_training(
            data_path=args.data_path,
            batch_size=args.batch_size
        )
        
        print(f"测试集大小: {test_dataset.get_dataset_size()}")
        
        # 基本评估
        print("\n🔍 基本模型评估...")
        basic_metrics = evaluate_model(model, test_dataset)
        
        print("基本指标:")
        for metric, value in basic_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 详细评估
        print("\n🔬 详细模型评估...")
        detailed_metrics, cm, roc_auc = detailed_evaluation(model, test_dataset, args.output_dir)
        
        # 与基线比较
        improvement = compare_with_baseline(detailed_metrics, args.output_dir)
        
        print(f"\n✅ 评估完成!")
        print(f"结果已保存到: {args.output_dir}")
        print(f"最终准确率: {detailed_metrics['accuracy']:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"❌ 评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 