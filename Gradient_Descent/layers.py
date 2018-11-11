"""
   该部分保存神经网络的层(实际上是该层使用的激活函数)，每个层的类包含前向反馈和反向传播两种方法
"""
import numpy as np
from functions import *


# Sigmoid层
class Sigmoid:
    def __init__(self):
        self.out = None   # 由于反向传播时sigmoid层仅需要它的输出，所以保存在属性中

    # 前向反馈
    def forward(self, x):
        out = sigmoid(x)
        self.out = out   # 将输出保存到实例变量中
        return out

    # 反向传播
    def backward(self, dout):  # 传进来上游的导数
        dx = dout * (1.0 -self.out) * self.out    # 上游导数与该sigmoid层导数的乘积
        return dx


# Affine层——主要保存加权和的偏导
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx