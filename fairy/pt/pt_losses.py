# -*- coding: utf-8 -*-
"""
@FileName    : losses.py.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-23 16:19
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import torch


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    @staticmethod
    def forward(self, x, y):
        loss = torch.mean((x - y)**2)
        return loss
