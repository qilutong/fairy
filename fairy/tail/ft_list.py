# -*- coding: utf-8 -*-
"""
@FileName    : ft_list.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-08 14:57
@Modify      : None
"""
from __future__ import absolute_import, division, print_function


def list_insert(raw_list, index, data):
    """
    根据索引在列表中插入新值，修改了索引为负数的情况，使之更符合直觉
    :param raw_list: 要修改的列表
    :param index: 索引
    :param data: 插入的数据
    :return:
    """
    # -1插入最后
    if index == -1:
        raw_list.append(data)
        return
    # 其余负数加1
    if index < 0:
        index += 1
    raw_list.insert(index, data)
