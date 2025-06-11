# 华为云MindSpore模型训练部署指南

本指南介绍如何将MindSpore糖尿病预测项目部署到华为云进行训练。

## 部署方案对比

| 方案 | 适用场景 | 成本 | 性能 | 易用性 | 推荐度 |
|------|----------|------|------|---------|---------|
| **ECS服务器** | 开发调试、小规模训练 | 中 | 中 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Notebook实例** | 交互式开发、实验 | 低 | 低 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **ModelArts平台** | 大规模训练、生产环境 | 高 | 高 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 方案一：ECS云服务器部署（推荐）

### 1.1 创建ECS实例

1. **登录华为云控制台**
   - 进入 `弹性云服务器 ECS`
   - 点击 `购买弹性云服务器`

2. **选择配置**
   ```
   区域: 华北-北京四 (推荐)
   规格: s6.xlarge.4 (4核16GB) 或更高
   镜像: Ubuntu 20.04 server 64bit
   系统盘: 100GB 高IO
   网络: 默认VPC和子网
   安全组: 开放22端口(SSH)
   带宽: 按需选择
   ```

3. **设置登录方式**
   - 密钥对登录（推荐）或密码登录
   - 记录服务器IP地址

### 1.2 本地部署准备

1. **确保数据文件就位**
   ```bash
   # 确保数据文件在项目根目录
   ls diabetes_prediction_dataset.csv
   ```

2. **上传项目到服务器**
   ```bash
   cd mindspore
   
   # 方式1: 使用密钥文件
   python scripts/upload_project.py \
       --server_ip YOUR_SERVER_IP \
       --username root \
       --key_file ~/.ssh/your-key.pem
   
   # 方式2: 使用密码
   python scripts/upload_project.py \
       --server_ip YOUR_SERVER_IP \
       --username root \
       --password YOUR_PASSWORD
   ```

### 1.3 服务器环境配置

1. **SSH登录服务器**
   ```bash
   ssh -i ~/.ssh/your-key.pem root@YOUR_SERVER_IP
   ```

2. **运行部署脚本**
   ```bash
   cd ~/mindspore_project
   bash scripts/deploy_to_ecs.sh
   ```

3. **激活Python环境**
   ```bash
   source mindspore_env/bin/activate
   ```

4. **上传数据文件**
   ```bash
   # 在本地运行，上传数据文件
   scp -i ~/.ssh/your-key.pem diabetes_prediction_dataset.csv root@YOUR_SERVER_IP:~/mindspore_project/
   ```

### 1.4 开始训练

1. **运行云端训练脚本**
   ```bash
   # 在服务器上运行
   cd ~/mindspore_project
   source mindspore_env/bin/activate
   
   # 开始训练
   python scripts/cloud_train.py \
       --data_file ./diabetes_prediction_dataset.csv \
       --epochs 100 \
       --batch_size 64 \
       --learning_rate 0.001 \
       --save_model \
       --output_dir ./output
   ```

2. **后台运行训练（推荐）**
   ```bash
   # 使用nohup后台运行
   nohup python scripts/cloud_train.py \
       --data_file ./diabetes_prediction_dataset.csv \
       --epochs 100 \
       --save_model \
       > training.log 2>&1 &
   
   # 查看训练日志
   tail -f training.log
   ```

3. **监控训练进度**
   ```bash
   # 查看GPU使用情况（如果有GPU）
   nvidia-smi
   
   # 查看系统资源
   htop
   
   # 查看训练日志
   tail -f output/logs/training_*.log
   ```

---

## 方案二：Notebook实例部署

### 2.1 创建Notebook实例

1. **进入ModelArts控制台**
   - 选择 `开发环境` → `Notebook`
   - 点击 `创建`

2. **配置Notebook**
   ```
   名称: mindspore-diabetes-notebook
   镜像: MindSpore 2.3.1-python3.9-ubuntu20.04
   规格: 通用计算型 (2核8GB) 或更高
   存储: 100GB
   ```

### 2.2 上传代码和数据

1. **启动Notebook**
   - 等待实例启动完成
   - 点击 `打开` 进入Jupyter环境

2. **上传项目文件**
   - 使用Jupyter界面上传项目文件
   - 或使用Terminal命令行操作

3. **安装依赖**
   ```bash
   # 在Notebook Terminal中运行
   pip install pandas scikit-learn matplotlib seaborn flask
   ```

### 2.3 运行训练

1. **创建训练Notebook**
   ```python
   # 在新的Notebook cell中
   import sys
   sys.path.append('.')
   
   # 运行训练脚本
   exec(open('scripts/cloud_train.py').read())
   ```

2. **交互式训练**
   ```python
   # 可以逐步执行训练过程，便于调试
   from config.model_config import ModelConfig
   from training.train import main
   
   # 设置参数并运行
   main()
   ```

---

## 方案三：ModelArts平台部署（生产级）

### 3.1 准备OBS存储

1. **创建OBS桶**
   ```
   桶名称: diabetes-model-bucket
   区域: 与训练作业相同区域
   存储类别: 标准存储
   ```

2. **创建目录结构**
   ```
   diabetes-model-bucket/
   ├── data/                 # 训练数据
   ├── code/                 # 训练代码
   ├── output/               # 输出结果
   └── logs/                 # 训练日志
   ```

### 3.2 上传数据和代码

1. **打包项目代码**
   ```bash
   # 创建代码压缩包
   cd mindspore
   zip -r mindspore_project.zip . -x "*.pyc" "__pycache__/*" ".git/*"
   ```

2. **上传到OBS**
   - 通过华为云控制台上传
   - 或使用OBS客户端工具

### 3.3 创建训练作业

1. **进入ModelArts控制台**
   - 选择 `训练管理` → `训练作业`
   - 点击 `创建`

2. **配置训练作业**
   ```
   作业名称: diabetes-mindspore-training
   算法来源: 自定义算法
   代码目录: obs://diabetes-model-bucket/code/
   启动文件: mindspore_project.zip/scripts/cloud_train.py
   数据来源: obs://diabetes-model-bucket/data/
   训练输出: obs://diabetes-model-bucket/output/
   
   资源配置:
   - 规格: modelarts.vm.cpu.8u (8核32GB)
   - 节点个数: 1
   
   运行参数:
   - data_file: /opt/ml/input/data/diabetes_prediction_dataset.csv
   - epochs: 100
   - batch_size: 64
   - save_model: true
   ```

3. **提交并监控训练**
   - 点击 `立即创建`
   - 在作业列表中查看训练状态
   - 点击作业名称查看详细日志

---

## 成本估算

### ECS方案成本
```
配置: s6.xlarge.4 (4核16GB)
价格: 约 1.2元/小时
训练时间: 2-4小时
总成本: 3-5元
```

### Notebook方案成本
```
配置: 通用计算型 (2核8GB)
价格: 约 0.8元/小时  
训练时间: 4-8小时
总成本: 3-6元
```

### ModelArts方案成本
```
配置: modelarts.vm.cpu.8u (8核32GB)
价格: 约 2.4元/小时
训练时间: 1-2小时
总成本: 2-5元
```

---

## 性能优化建议

### 1. 数据预处理优化
```python
# 在训练脚本中增加数据缓存
processor.cache_processed_data = True
```

### 2. 模型配置优化
```python
# 针对云端环境优化
training_config = TrainingConfig(
    batch_size=128,          # 增大批次大小
    learning_rate=0.002,     # 适当增大学习率
    num_workers=4,           # 多进程数据加载
    prefetch_factor=2        # 数据预加载
)
```

### 3. 监控和日志
```bash
# 实时监控资源使用
watch -n 1 'free -h && df -h'

# 查看训练日志
tail -f output/logs/training_*.log
```

---

## 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   # 更新pip和setuptools
   pip install --upgrade pip setuptools
   
   # 使用国内镜像源
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mindspore
   ```

2. **内存不足**
   ```python
   # 减小批次大小
   training_config.batch_size = 32
   
   # 启用梯度累积
   training_config.gradient_accumulation_steps = 2
   ```

3. **网络连接问题**
   ```bash
   # 检查安全组设置
   # 确保开放必要端口（SSH: 22）
   ```

### 联系支持

如遇到问题，可以：
1. 查看华为云官方文档
2. 提交工单获取技术支持
3. 参考社区论坛解决方案

---

## 总结

推荐使用**ECS方案**进行初次部署，具有以下优势：
- ✅ 部署简单，学习成本低
- ✅ 成本可控，按需付费
- ✅ 完全控制训练环境
- ✅ 便于调试和优化

如需更高性能或生产级别的训练，可考虑升级到**ModelArts平台**。 