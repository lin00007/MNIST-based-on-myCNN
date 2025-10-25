import os
import sys           # 处理文件路径
import torch         # PyTorch深度学习框架核心库
import torch.nn as nn         # PyTorch的神经网络模块
import torch.optim as optim   # 优化器
from torchvision import datasets, transforms  # 计算机视觉数据集和图像预处理工具
from torch.utils.data import DataLoader       # 批量加载数据的工具
from tqdm import tqdm                         # 进度条显示工具
from matplotlib import pyplot as plt          # 数据可视化库
from SampleCNN import SampleCNN               # 导入自定义的卷积神经网络模型
from torchsummary import summary              # 用于打印模型摘要信息（单纯用于展示，可省略）

def main(): 
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    print(f"Using device: {device}")

    # 数据预处理：把图片转成张量 + 标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))      #把0-1变成均值为0，方差为1的正态分布，可以使神经网络快速收敛
    ])

    # 加载MNIST数据集
    data_path = './data'
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 把训练集分成64张/批，每次训练打乱顺序（避免过拟合）
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Data loaded successfully.")

    net = SampleCNN().to(device) # 加载模型，放到设备上
    summary(net, (1, 28, 28))  # 打印模型结构，输入是单通道，像素28*28（单纯用于展示，可省略）

    loss_function = nn.CrossEntropyLoss()  #交叉熵损失函数，适用于多分类问题
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam优化器

    epoches = 20           #训练轮次
    best_acc = 0.0
    save_path = './results/weights/SimpleCNN'            #权重路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 记录每个epoch的损失和准确率
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epoches):
        net.train()                                 # 训练模式（有些层比如Dropout只在训练时生效）
        acc_num = torch.zeros(1).to(device)         # 创建包含1个元素、值为0的张量（防止使用int累加时每次要在CPU和GPU上来回切换），累计预测对的数量      
        sample_num = 0
        train_loss = 0.0                            # 记录训练中的总损失
         
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)     # 用 tqdm 显示进度条，遍历每一批训练数据
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)      # 把数据放到设备上
            sample_num += images.size(0)
            optimizer.zero_grad()                                      # 1. 清空之前的梯度
            outputs = net(images)                                      # 2. 模型预测：输入图片，输出10个概率（对应0-9）
            loss = loss_function(outputs, labels)                      # 3. 计算损失
            loss.backward()                                            # 4. 反向传播：计算梯度（告诉模型哪里错了、错多少）
            optimizer.step()                                           # 5. 优化器更新参数：根据梯度调整模型参数

            train_loss += loss.item()                                  # 累计损失
            _, predicted = torch.max(outputs.data, 1)                  # 取概率最大的类别作为预测结果
            acc_num += (predicted == labels).sum().item()              # 统计预测对的数量

            train_bar.set_description(f"Epoch [{epoch+1}/{epoches}]")
        # 计算这一轮的平均损失和准确率
        train_loss_avg = train_loss / len(train_loader.dataset)
        train_acc = acc_num.item() / sample_num
        # 存储结果，后面绘图
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        print(f"Epoch [{epoch+1}/{epoches}] Training Loss: {train_loss_avg:.4f}, Training Acc: {train_acc:.4f}")
        
        #每轮训练后，用测试集评估效果
        net.eval()                 # 评估模式
        acc_num = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                acc_num += torch.sum(predicted == labels)
        
        val_acc = acc_num.item() / len(test_dataset)  # 计算测试集准确率
        val_accuracies.append(val_acc)
        # 保存最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(save_path, 'best_model.pth'))
        print(f"Best model saved with accuracy: {best_acc:.4f}")

    epochs_range = range(1, epoches + 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()

    if not os.path.exists('./results/figures'):
        os.makedirs('./results/figures')
    plt.savefig('./results/figures/training_validation_plots.png')
    plt.show()

if __name__ == '__main__':
    main()
