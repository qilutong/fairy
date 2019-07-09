# -*- coding: utf-8 -*-
"""
@FileName    : ft_dict.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-07-09 下午4:21
@Modify      : None
"""
from __future__ import absolute_import, division, print_function
import copy


def merge_key(data):
    """
    合并两个字典中关键字相同的部分
    :param data:
    :return:
    """
    new_data = {}
    tmp_list = []  # 临时存储value的值
    keys = list(data[0].keys())

    for key in keys:
        for dict_tmp in data:
            value = dict_tmp[key]
            tmp_list.append(value)
        new_data[key] = copy.copy(tmp_list)
        print(new_data)
        tmp_list.clear()  # 清除缓存

    return new_data
