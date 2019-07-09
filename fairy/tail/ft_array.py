# -*- coding: utf-8 -*-
"""
@FileName    : ft_array.py
@Description : 数组相关操作
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-07 17:54
@Modify      : None
"""
from __future__ import absolute_import, division, print_function
import numpy as np

from .ft_list import list_insert


def array_merge(array_list, axis):
    """
    将numpy数组列表按某一维度拼接在一起
    :param array_list: Numpy 数组列表
    :param axis: 要合并的轴
    :return: Numpy 数组
    """
    # print(array_list[0].shape)
    # print(array_list[1].shape)
    return np.concatenate(tuple(array_list), axis=axis)


def add_dim(data, axis):
    """
    给numpy数组增加新维度
    :param data: Numpy 数组
    :param axis: 要增加的维度索引
    :return: 新 Numpy 数组
    """
    size = list(data.shape)
    list_insert(size, axis, 1)
    return data.reshape(size)
