# -*- coding: utf-8 -*-
"""
@FileName    : simple_net.py
@Description : 使用numpy构建一个简单的神经网络
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-09-03 22:28
@Modify      : None
"""
from __future__ import absolute_import, division, print_function
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机输入输出数据
x = np.random.randn(N, D_in)  # (64, 1000)
y = np.random.randn(N, D_out)  # (64, 10)

# 随机初始化权重
w1 = np.random.randn(D_in, H)  # (1000, 100)
w2 = np.random.randn(H, D_out)  # (100, 10)

# 学习率
learning_rate = 1e-6
for t in range(500):
    # 前项传播：计算预测 y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算并输出 loss
    loss = np.square(y_pred - y).sum()

    if t % 10 == 0:
        print(t, loss)

    # 使用反向传播通过 loss 计算 w1 和 w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
