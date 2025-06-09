# cifar10_classification.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据加载和预处理
def load_data(batch_size=128):
    # 增加更多的数据增强方法
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

# 定义带残差连接的模型
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 32
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 增加残差块的数量
        self.layer1 = self._make_layer(32, 3, stride=1)
        self.layer2 = self._make_layer(64, 3, stride=2)
        self.layer3 = self._make_layer(128, 3, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 训练模型
def train_model(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=30, device='cuda'):
    model.to(device)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(trainloader)
        train_accuracy = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 评估模型
        test_loss, test_accuracy = evaluate_model(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)
    
    return model, best_accuracy

# 评估模型
def evaluate_model(model, testloader, criterion, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(testloader)
    test_accuracy = 100.0 * correct / total
    
    return test_loss, test_accuracy

# 绘制训练曲线
def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

# 可视化卷积核
def visualize_filters(model, layer_name, num_filters=16):
    # 获取指定层的权重
    layer = getattr(model, layer_name)
    if isinstance(layer, nn.Conv2d):
        filters = layer.weight.data.cpu().numpy()
        
        # 确保只显示前几个滤波器
        num_filters = min(num_filters, filters.shape[0])
        
        plt.figure(figsize=(10, 10))
        for i in range(num_filters):
            filter_img = filters[i, 0]  # 获取第一个通道的滤波器
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
            
            plt.subplot(4, 4, i+1)
            plt.imshow(filter_img, cmap='viridis')
            plt.axis('off')
            plt.title(f'Filter {i+1}')
        
        plt.tight_layout()
        plt.savefig(f'conv_filters_{layer_name}.png')
        plt.close()

# 可视化特征图
def visualize_feature_maps(model, sample_image, layer_name):
    model.eval()
    
    # 创建一个钩子来获取中间层的输出
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子
    layer = getattr(model, layer_name)
    hook = layer.register_forward_hook(get_activation(layer_name))
    
    # 前向传播
    with torch.no_grad():
        model(sample_image.unsqueeze(0).to(device))
    
    # 释放钩子
    hook.remove()
    
    # 获取激活值
    feature_maps = activation[layer_name].cpu().numpy()[0]
    
    # 可视化特征图
    num_maps = min(16, feature_maps.shape[0])
    plt.figure(figsize=(10, 10))
    for i in range(num_maps):
        plt.subplot(4, 4, i+1)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Feature Map {i+1}')
    
    plt.tight_layout()
    plt.savefig(f'feature_maps_{layer_name}.png')
    plt.close()

# 主函数
def main():
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    trainloader, testloader, classes = load_data()
    
    # 创建模型
    model = ResNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度器
    
    # 训练模型
    start_time = time.time()
    model, best_accuracy = train_model(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=30, device=device)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    
    # 可视化卷积核
    visualize_filters(model, 'conv1')
    
    # 可视化特征图
    sample_image, _ = next(iter(testloader))
    visualize_feature_maps(model, sample_image[0], 'layer1')

if __name__ == "__main__":
    main()