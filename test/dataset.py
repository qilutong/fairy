# -*- coding: utf-8 -*-
"""
@FileName    : dataset.py
@Description : dataset测试
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-21 10:47
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import fairy
import numpy as np

a = [i for i in range(10)]  # 列表
b = np.arange(15).reshape(5, 3)  # numpy数组
c = {"a": [1, 2, 3], "b": [4, 5, 6]}  # 字典类型
d = ([i for i in range(10)], [i for i in range(10, 20)])  # 元组类型

# data1 = fairy.data.dataset(a, shuffle=True)
# data2 = fairy.data.dataset(b, shuffle=True)
# data3 = fairy.data.dataset(c, shuffle=True)
# data4 = fairy.data.dataset(d, shuffle=True)


def fuc1(x):
    return x+1


data1 = fairy.data.Dataset(a).repeat(2).shuffle().map(fuc1).make_iterator()
# data1 = fairy.data.Dataset(a).repeat(2).shuffle()
data2 = fairy.data.Dataset(b).repeat().shuffle().make_iterator()
data3 = fairy.data.Dataset(c).repeat().shuffle().make_iterator()
data4 = fairy.data.Dataset(d).repeat().shuffle().make_iterator()


# for i in range(30):
#     print(data1.get_next())

for i in data1:
    print(i)
for i in range(20):
    print(next(data2))
for i in range(10):
    print(next(data3))
for i in range(30):
    print(next(data4))


