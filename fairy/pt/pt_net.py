# -*- coding: utf-8 -*-
"""
@FileName    : net.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-10 14:32
@Modify      : None
"""
from __future__ import absolute_import, division, print_function
import abc
import torch


class Net(torch.nn.Module):
    """
    pytorch 网络基础类（抽象类）

    Attributes:
        ring_list: 列表
        index: 索引
        length: 列表长度
        counter: 循环计数器，表示已经循环了几次，调用 next()函数自动计数
    """

    @abc.abstractmethod # 定义抽象方法，无需实现功能
    def read(self):
        """子类必须定义读功能"""
        pass




