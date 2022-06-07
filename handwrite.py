# Filename: handwrite.py
# Description: 手写数字识别演示Demo，按住鼠标左键绘制，键盘Enter识别，Backspace删除绘图
# Author: Denis
# Date: 2022-06-07 @ sec-chip
# Github: www.github.com/oslomayor
# Update:
# Version 1

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

global start_x, start_y, isdraw, fig, axs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 网络模型
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


def read_image(img_name):
    # to do: 绘图处理成tensor传给模型
    # return torch.rand(28, 28)
    img = Image.open(img_name)
    out1 = img.convert('L')  # color (32 bit) to gray (8 bit)
    out1.save(img_name+'_out1.png')
    out2 = out1.resize((28, 28))
    out2.save(img_name+'_out2.png')
    x = np.array(out2).reshape(1, 28, 28)
    x = torch.from_numpy(x)
    # 训练时，torch 的 transform 默认自动归一化，255转成0.9922， 除以257刚好差不多
    x = (255 - x) / 257
    return x


def on_key_press(event):
    global start_x, start_y, isdraw, fig, axs
    img_name = './handwriting.png'
    print(f'{event.key} is pressed.')
    if event.key == 'enter':
        # 保持整个figure
        # plt.savefig(img_name)
        # 以下2行代码，只保存第一个子窗口
        # ref: https://newbedev.com/save-a-subplot-in-matplotlib
        extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(img_name, bbox_inches=extent)
        img_array = read_image(img_name)
        pred_num = predict(img_array)
        print(f'the number {pred_num} is writen')
        axs[1].text(0.25, 0.25, pred_num, {'size': 120})
        axs[1].figure.canvas.draw()
    elif event.key == 'backspace':
        axs[0].clear()
        axs[0].set(xlim=(0, 1), ylim=(0, 1))
        axs[0].set(xticks=[0, 1], yticks=[0, 1])
        axs[0].set_title('Input')
        axs[0].set_aspect('equal', 'box')  # 坐标比例1:1，方形
        axs[0].figure.canvas.draw()
        axs[1].clear()
        axs[1].set(xlim=(0, 1), ylim=(0, 1))
        axs[1].set(xticks=[0, 1], yticks=[0, 1])
        axs[1].set_title('Predict')
        axs[1].set_aspect('equal', 'box')
        axs[1].figure.canvas.draw()


def on_mouse_press(event):
    global start_x, start_y, isdraw, fig, axs
    isdraw = True
    start_x = event.xdata
    start_y = event.ydata
    print(f'press: {start_x}, {start_y}')


def on_mouse_move(event):
    global start_x, start_y, isdraw, fig, axs
    if isdraw:
        end_x = event.xdata
        end_y = event.ydata
        x1 = [start_x, end_x]
        y1 = [start_y, end_y]
        axs[0].plot(x1, y1, color='black', linestyle='-', linewidth='20')
        axs[0].figure.canvas.draw()
        start_x = end_x
        start_y = end_y


def on_mouse_release(event):
    global start_x, start_y, isdraw, fig, axs
    print('release: ', event.xdata, event.ydata, isdraw)
    isdraw = False


def predict(img_array):
        output = model(img_array)
        pred = output.max(1, keepdim=True)[1].item()
        return pred
        # print('input img array:\n', img_array)


def fig_init():
    global start_x, start_y, isdraw, fig, axs
    isdraw = False
    start_x, start_y = 0, 0
    fig, axs = plt.subplots(1, 2)  # 返回 fig 和 axs 列表, axs[0,1]

    axs[0].set(xlim=(0, 1), ylim=(0, 1))
    axs[0].set(xticks=[0, 1], yticks=[0, 1])
    axs[0].set_title('Input')
    axs[0].set_aspect('equal', 'box')  # 坐标比例1:1，方形
    axs[1].set(xlim=(0, 1), ylim=(0, 1))
    axs[1].set(xticks=[0, 1], yticks=[0, 1])
    axs[1].set_title('Predict')
    axs[1].set_aspect('equal', 'box')

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    # plt.axis([0, 1, 0, 1])
    plt.show()


if __name__ == '__main__':
    # 加载训练好的模型
    model_name = './models/mnist_cnn_batch_size_64_epochs_5_accuracy_99_14.pkl'
    if os.path.exists(model_name):
        print('using model: ', model_name)
        model = torch.load(model_name)
        model.eval()
    else:
        print('''can't find model''', model_name)
    # GUI界面初始化
    fig_init()
