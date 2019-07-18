# -*- coding: utf-8 -*-
"""
@FileName    : tf_data.py
@Description : 使用tf原生API进行数据处理
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-16 15:45
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf


def tf_dataset(data,
               shuffle=False,
               buffer_size=None,
               func=None,
               batch_size=1,
               repeat=None):
    """
    数据集处理，基于tf.data.Dataset
    :param data: 数组 [1,2,3,...];a
                 元组 (x_data, y_data);
                 字典 {"x":x_data, "y":y_data}
    :param shuffle: 是否进行shuffle
    :param buffer_size: 当shuffle=True时必须传入，推荐等于数据个数
    :param func: 数据处理函数，输入输出格式与data相同
    :param batch_size: batch大小，默认为1
    :param repeat: 循环几次，默认无限循环
    :return: 数据迭代器，使用get_next()
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        if buffer_size is None:
            raise ValueError("Parameter buffer_size is undefined")
        # shuffle在数据处理之前，如：混洗文件名而不是读取后的数组
        dataset = dataset.shuffle(buffer_size=buffer_size)

    if func is not None:
        dataset = dataset.map(func)

    dataset = dataset.batch(batch_size)
    if repeat is None:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(repeat)

    iterator = dataset.make_initializable_iterator()
    return iterator
