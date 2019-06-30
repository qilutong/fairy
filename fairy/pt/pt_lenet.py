# -*- coding: utf-8 -*-
"""
@FileName    : lenet.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-13 16:51
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    pytorch 网络基础类（抽象类）

    Attributes:

    """

    def __init__(self, ):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 6, (5, 5))  # output (N, C_{out}, H_{out}, W_{out})`
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),
                         (2, 2))  # F.max_pool2d的返回值是一个Variable
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
