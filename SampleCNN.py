from torch import nn                        # 导入PyTorch的神经网络模块
import torch.nn.functional as F             # 导入PyTorch的函数式API（包含激活函数、池化等操作）

class SampleCNN(nn.Module):                 # 自定义CNN类，继承PyTorch所有神经网络的基类nn.Module
    def __init__(self):
        super(SampleCNN, self).__init__()   # 调用父类nn.Module的初始化方法
        # 第一层卷积层，输入通道数为1（灰度图，若是彩色图则为3），输出通道为5（自定义，此处用5个不同的3×3卷积核，提取5种不同的特征），卷积核大小为3*3，填充为1（图像大小不变）
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        # 第二层卷积层，输入通道数为5（和上一层的输出通道数一致），输出通道为5，卷积核大小为3*3，填充为1
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        # 最大值池化层（用于压缩特征图），池化窗口大小为2*2
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # 将特征展平为一维向量（为全连接层做准备）
        self.flatten = nn.Flatten()
        # 全连接层，245为输入特征数，10为输出特征数
        self.fc = nn.Linear(245, 10)  

    def forward(self, x):   # 前向传播方法，定义数据如何通过网络流动；是PyTorch的核心方法，当调用模型时（如model(x)），会自动执行forward函数
        x = F.relu(self.conv1(x))      # 输入x经过conv1卷积后，再通过ReLU激活函数（负数变 0，正数不变，给网络加入 “非线性”），输出形状为（批量，5，28，28）
        x = self.maxpool(x)            # 经过最大值池化层，14*14*5
        # 第二层卷积层 + ReLU激活函数 + 最大值池化
        x = F.relu(self.conv2(x))      # 14*14*5
        x = self.maxpool(x)            # 7*7*5
        x = self.flatten(x)            # 展平为一维向量
        x = self.fc(x)                 # 全连接层输出类别分数
        x = F.log_softmax(x, dim=1)    # log_softmax 归一化，输出概率分布
        return x 