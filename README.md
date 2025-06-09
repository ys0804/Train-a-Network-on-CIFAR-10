# CIFAR-10 图像分类项目
## 简介
这个项目实现了基于 PyTorch 的 CIFAR-10 图像分类任务，采用了带有残差连接的卷积神经网络 (ResNet) 架构。项目包含数据增强、模型训练、评估和可视化等完整流程。
## 项目特点
使用 ResNet 架构，具有更好的训练效果和泛化能力
实现了多种数据增强技术，包括随机裁剪、水平翻转、旋转和颜色抖动
包含训练过程可视化、卷积核可视化和特征图可视化功能
提供了详细的训练日志和模型性能评估
环境要求
Python 3.7+
PyTorch 1.7+
torchvision
matplotlib
numpy
tqdm

可以使用以下命令安装所需依赖：

bash
pip install torch torchvision matplotlib numpy tqdm
## 代码结构
主要功能模块：

数据加载和预处理：load_data()
模型定义：ResidualBlock和ResNet类
模型训练：train_model()
模型评估：evaluate_model()
训练曲线绘制：plot_training_curves()
卷积核可视化：visualize_filters()
特征图可视化：visualize_feature_maps()
### 使用方法
克隆项目代码：

bash
git clone https://github.com/your_username/cifar10_classification.git
cd cifar10_classification

运行训练和评估脚本：

bash
python cifar10_classification.py

### 查看训练结果和可视化图像：
训练过程中的损失和准确率曲线：training_curves.png
卷积核可视化结果：conv_filters_conv1.png
特征图可视化结果：feature_maps_layer1.png
最佳模型权重：best_model.pth
## 模型架构
本项目使用的 ResNet 架构包含：

3 个残差块层
自适应平均池化层
全连接分类层
总共约 110 万个参数
训练配置
优化器：Adam
初始学习率：0.001
学习率调度：每 10 个 epoch 衰减为原来的 0.1 倍
损失函数：交叉熵损失
训练轮数：30
批大小：128
##可视化功能
###项目提供了两种可视化功能：

卷积核可视化：展示第一层卷积层的滤波器
特征图可视化：展示样本图像通过网络后第一层残差块的特征图

这些可视化有助于理解模型学习到的特征和表示。
