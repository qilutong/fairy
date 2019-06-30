# -*- coding: utf-8 -*-
"""
@FileName    : dataset.py
@Description : 数据集类
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-28 15:11
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import numpy as np

import fairy.tail as ft

from .ring import Ring


class Dataset(object):
    """
    Dataset类，可用来对数据集进行处理和迭代

    Attributes:
        ring: 循环列表类
        counter: 循环计数器，表示已经循环了几次
        batch_list: batch数据缓存区
        batch_size: batch大小，默认为1
        epochs: 数据循环次数，默认为1
    """

    def __init__(self, data, batch_size=1, epochs=1, shuffle=False):
        """
        初始化
        :param data: 数组 x_data
                     元组 (x_data, y_data,...)
                     字典 {"x":x_data, "y":y_data,...}
        """
        self.ring = Ring(ft.data_zip(data))

        self.counter = self.ring.get_counter()

        self.batch_list = []

        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_flag = shuffle

    def shuffle(self, flag=True):
        """
        对数据进行shuffle处理
        :return:self
        """
        if flag:
            self.shuffle_flag = True
            np.random.shuffle(self.ring.data)
        else:
            self.shuffle_flag = False
        return self

    def map(self, func):
        """
        对数据进行map处理
        :param func:处理函数，输入输出数据格式应该一致，如：add1(data)->data+1
        :return:self
        """
        """"""
        self.ring.data = [i for i in map(func, self.ring.data)]
        return self

    def batch(self, batch_size=1):
        """
        设置 batch_size 大小
        :param batch_size:批次大小，默认为1
        :return:self
        """
        self.batch_size = batch_size
        return self

    def repeat(self, epochs=None):
        """
        设置数据重复次数，默认None无限循环
        :param epochs:重复次数
        :return: self
        """
        if epochs is None:
            self.epochs = -1
        else:
            self.epochs = epochs
        return self

    def get_next(self):
        """
        获取一个 mini batch
        :return: numpy数组
        """
        # 暂存batch_size和counter
        batch_size = self.batch_size
        counter = self.counter

        if self.epochs == self.counter:
            raise StopIteration("Data has been iteratively completed")
        else:
            # 清空list
            self.batch_list.clear()
            # 最后一次epoch并且剩余不足batch_size时，将batch_size的值设为剩余元素数量
            if ((self.epochs - self.counter) == 1) and (
               (self.ring.length - self.ring.index) // self.batch_size == 0):
                batch_size = (
                    self.ring.length - self.ring.index) % self.batch_size

            for _ in range(batch_size):
                # 每个epoch混洗一次
                if (counter != self.counter) and self.shuffle_flag:
                    np.random.shuffle(self.ring.data)
                    counter = self.counter

                image_np = self.ring.next()
                self.counter = self.ring.get_counter()  # 更新self.counter
                image_np = np.array(image_np)

                image_np = ft.add_dim(image_np, 0)
                self.batch_list.append(image_np)

            return ft.array_merge(self.batch_list, axis=0)

    def make_iterator(self):
        """
        生成迭代器，迭代设置好的循环
        :return: 生成器
        """
        if self.epochs == self.counter:
            raise StopIteration("Data has been iteratively completed")
        else:
            while 1:
                yield self.get_next()


class DataIter(object):
    def __init__(self):
        pass

    def get_next(self):
        pass
