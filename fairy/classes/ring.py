# -*- coding: utf-8 -*-
"""
@FileName    : ring.py
@Description : 循环类
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-08 18: 59
@Modify      : None
"""
from __future__ import absolute_import, division, print_function


class Ring(object):
    """
    环形列表

    Attributes:
        data: 列表形式的数据，内部元素可以是元组、字典、数组、字符串...
        index: 索引
        length: 列表长度
        counter: 循环计数器，表示已经循环了几次，调用 next()函数自动计数
    """
    def __init__(self, raw_list):
        """初始化环形列表，输入为一个标准列表"""
        self.data = raw_list
        self.index = 0
        self.length = len(raw_list)
        self.counter = 0

    def reset(self):
        """重置索引和循环计数器"""
        self.index = 0
        self.counter = 0

    def set(self, index, value):
        """设置对应索引的值"""
        self.data[index] = value

    def set_index(self, index):
        """设置当前索引"""
        self.index = index % self.length

    def set_data(self, new_list):
        """设置新列表"""
        self.data = new_list
        self.reset()

    def set_counter(self, counter):
        """设置当前循环计数器"""
        self.counter = counter

    def get(self, index=None):
        """获取索引对应的值，默认获取当前索引的值"""
        if index is None:
            index = self.index
        return self.data[index % self.length]

    def get_index(self):
        """获取当前索引"""
        return self.index

    def get_list(self):
        """获取当前循环的列表"""
        return self.data

    def get_counter(self):
        """获取当前循环计数器"""
        return self.counter

    def next(self):
        """获取当前索引的值并将索引+1"""
        if self.index < self.length - 1:
            self.index += 1
        # 末尾指向开头，循环计数加一
        elif self.index == self.length - 1:
            self.index = 0
            self.counter += 1
        else:
            raise ValueError("Index Out Of Bounds")

        return self.data[self.index - 1]
