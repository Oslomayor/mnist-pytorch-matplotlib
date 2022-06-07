# Filename: mnist.py
# Description: 训练手写数字识别神经网络模型，保存模型，测试正确率
# Author: Denis
# Date: 2022-06-07 @ sec-chip
# Github: www.github.com/oslomayor
# Update:
# Version 1

# 1 加载库
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 2 定义超参数
BATCH_SIZE = 64  # 每批处理的数据量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5  # 训练的轮数

# 3 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转成tensor
    # 似乎默认自动归一化
    # transforms.Normalize((0.1307,), (0.3081,))  # 正则化，降低模型复杂度
])

# 4 下载、加载数据
# 如果手动从MNIST官网下载数据集，需要注意以下：
# 1. torchvision内部固定的目录结构为：MNIST/raw/(数据集)
# 2. 例如把数据集存放在路径：./dataset/MNIST/raw/(数据集)，则 root='./dataset/'
# 3. 还要注意数据集文件名：MNIST官网手动下载的数据集名称和torchvision定义不同，
#    所有后缀.ubyte改为-ubyte，例如把 train-images-idx3.ubyte 改为 train-images-idx3-ubyte
train_set = datasets.MNIST(root='./dataset/', train=True, download=False, transform=transform)
test_set = datasets.MNIST(root='./dataset/', train=False, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 显示 mnist 图像
def display():
    with open('./dataset/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
        file = f.read()
    img_num = int.from_bytes(file[4:8], 'big')  # 图像总数 60000
    img_row = int.from_bytes(file[8:12], 'big') # 图像行数 28
    img_col = int.from_bytes(file[12:16], 'big') # 图像列数 28
    img_cnt = 0
    pixel_num = img_row*img_col*100  # *img_num
    pixels = []
    for i in range(pixel_num):
        pixels.append(int.from_bytes(file[16+i:16+1+i], 'big'))
    pixels = np.array(pixels)
    imgs = pixels.reshape([100, 28, 28])  # ([img_num, 28, 28])
    # 显示第1张图片
    plt.imshow(imgs[0, :, :], cmap='gray')
    # 显示第2张图片
    # plt.imshow(imgs[1, :, :], cmap='gray')
    plt.show()


# 5 构建网络模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # in, out, kernel
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)

    # 前向计算
    def forward(self, x):
        input_size = x.size(0)      # batch_size
        x = self.conv1(x)           # in: batch*1*28*28, out: batch*10*24*24 (24=28-5+1)
        x = F.relu(x)               # keep shape, out: batch*10*24*24
        x = F.max_pool2d(x, 2, 2)   # out: batch*10*12*12
        x = self.conv2(x)           # out: batch*20*10*10 (10=12-3+1)
        x = F.relu(x)               # keep shape, out: batch*20*10*10
        x = x.view(input_size, -1)  # 拉平， -1 自动计算维度 2000=20*10*10
        x = self.fc1(x)             # in: batch*2000, out: batch*500
        x = F.relu(x)               # keep shape
        x = self.fc2(x)             # out: batch*10
        output = F.log_softmax(x, dim=1)  # 计算每个数字的概率
        return output


# 6 定义优化器
model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# 7 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到device
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 1000 == 0:
            print("Train Epoch: {}  batch: {}  Loss: {:.6f}".format(epoch, batch_index, loss.item()))


# 8 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            # 部署到device
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.max(1, keepdim=True)[1]
            # 比较 pred 与 target 的相同值个数，统计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test Avg loss: {:.4f}, Accuracy: {:.3f}%\n".format(test_loss, 100.0*correct/len(test_loader.dataset)))


if __name__ == '__main__':
    model_name = './models/mnist_cnn_1.pkl'  # 模型保持路径，需要预先在代码所在目录下建立models文件夹
    # display()  # 显示 mnist 图片
    if os.path.exists(model_name):
        model = torch.load(model_name)
        test_model(model, DEVICE, test_loader)
    else:
        # 9 调用方法 7/8
        for epoch in range(1, EPOCHS + 1):
            train_model(model, DEVICE, train_loader, optimizer, epoch)
            test_model(model, DEVICE, test_loader)
        torch.save(model, model_name)
        print(f'model saved to {model_name}\n')
    print('Finished \n')

