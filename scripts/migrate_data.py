"""
数据迁移脚本 - 从原XGBoost项目迁移数据
"""
import os
import shutil
import pandas as pd
from pathlib import Path

def migrate_data():
    """迁移数据文件"""
    print("📊 开始数据迁移...")
    
    # 源数据路径
    source_data = "../diabetes_prediction_dataset.csv"
    
    # 目标路径
    target_dir = "./data"
    os.makedirs(target_dir, exist_ok=True)
    target_data = os.path.join(target_dir, "diabetes_prediction_dataset.csv")
    
    # 复制数据文件
    if os.path.exists(source_data):
        shutil.copy2(source_data, target_data)
        print(f"✅ 数据文件已复制到: {target_data}")
        
        # 验证数据
        data = pd.read_csv(target_data)
        print(f"数据形状: {data.shape}")
        print(f"特征列: {list(data.columns)}")
        
    else:
        print(f"❌ 源数据文件不存在: {source_data}")
    
    print("✅ 数据迁移完成!")

if __name__ == "__main__":
    migrate_data() 