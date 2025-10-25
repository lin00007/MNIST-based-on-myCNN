import torch                                        # PyTorch核心库
import torchvision.transforms as transforms         # 图像预处理工具
from torchvision import datasets                    # 内置数据集
import matplotlib.pyplot as plt                     # 可视化工具
from SampleCNN import SampleCNN                     # 导入自定义的卷积神经网络模型

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),                          # 转成张量
    transforms.Normalize((0.1307,),(0.3081,))       # 标准化
])
# 加载数据集
data_path = './data'
val_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)    
# 加载模型
model = SampleCNN()
model.load_state_dict(torch.load('results/weights/SimpleCNN/best_model.pth'))   # 加载训练时保存的最佳权重
# 设置模型为评估模式
model.eval()
# 预测10张图片
plt.figure(figsize=(10,5))                       # 创建画布
for i in range(10):
    image, label = val_dataset[i][0], val_dataset[i][1]
    with torch.no_grad():                        # 关闭梯度计算
        output = model(image.unsqueeze(0))       # 增加batch维度，模型输入需要[batch_size, 通道, 高, 宽]
    # 获得预测结果
    _, predicted = torch.max(output, 1)          # 在第1维（类别维度）取最大值，返回概率最大值和对应的索引
    predicted_label = predicted.item()           # 将张量转为Python数值
    # 绘制图片
    plt.subplot(2,5,i+1)                         # 2行5列的子图，第i+1个位置
    plt.imshow(image.squeeze().numpy(), cmap='gray')   # 去掉多余的维度（从[1,28,28]变回[28,28]），再转成numpy数组用于显示
    plt.title(f'Predicted:{predicted_label},True:{label}')
    plt.axis('off')

plt.tight_layout()                               # 自动调整子图间距，避免重叠
plt.show()
