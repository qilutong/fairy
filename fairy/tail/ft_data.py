# -*- coding: utf-8 -*-
"""
@FileName    : ft_data.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-21 14:41
@Modify      : None
"""
from __future__ import absolute_import, division, print_function


def data_zip(data):
    """
    输入数据，返回一个拼接了子项的列表，如([1,2,3], [4,5,6]) -> [[1,4], [2,5], [3,6]]
    {"a":[1,2],"b":[3,4]} -> [{"a":1,"b":3}, {"a":2,"b":4}]
    :param data: 数组 data
                 元组 (x, y,...)
                 字典 {"a":data1, "b":data2,...}
    :return: 列表或数组
    """
    if isinstance(data, tuple):
        return [list(d) for d in zip(*data)]
    if isinstance(data, dict):
        data_list = []
        keys = data.keys()

        for i in range(len(data[list(keys)[0]])):  # 迭代字典值中的数据
            data_dict = {}
            for key in keys:
                data_dict[key] = data[key][i]
            data_list.append(data_dict)

        return data_list

    return data
