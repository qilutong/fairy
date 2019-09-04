# -*- coding: utf-8 -*-
"""
@FileName    : pt_base.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-09-03 22:12
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

gpu = torch.cuda.is_available()

print("PyTorch version: ", torch.__version__)
if gpu:
    print("CUDA version: ", torch.version.cuda)
    print("cuDNN version: ", torch.backends.cudnn.version())
    print("GPU: ", torch.cuda.get_device_name())

# 设备
device = torch.device('cuda:0' if gpu else 'cpu')
"""
超参数
"""
EPOCHS = 5
NUM_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
"""
数据处理
"""
train_dataset = torchvision.datasets.MNIST(
    root='/home/hello/notebook/05-深度学习/PyTorch/data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_dataset = torchvision.datasets.MNIST(
    root='/home/hello/notebook/05-深度学习/PyTorch/data/',
    train=False,
    transform=transforms.ToTensor())
"""
Data loader
"""
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


class ConvNet(nn.Module):
    """
    简单卷积网络
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# 实例化，并分配设备
model = ConvNet(NUM_CLASSES).to(device)

# 定义Loss 和 optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        # 分配设备
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播并且优化参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, EPOCHS, i + 1, total_step, loss.item()))

# 测试
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))
