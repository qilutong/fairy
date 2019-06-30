# -*- coding: utf-8 -*-
"""
@FileName    : data_base.py
@Description : 基本数据操作
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-07 11:31
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from PIL import Image

import fairy.tail as ft
from fairy.classes import Ring


def open_image(image_name, mode, size=None):
    """
    打开指定图像
    :param image_name:文件名
    :param mode:打开模式
    :param size:size
    :return:PIL.Image.Image 对象
    """
    if mode is None:
        image = Image.open(image_name)
    elif mode == "RGB":
        image = Image.open(image_name).convert("RGB")
    elif mode == "GRAY":
        image = Image.open(image_name).convert("L")
    else:
        raise ValueError("mode value should be RGB or GRAY")

    # 修改size
    if size is not None:
        image = image.resize(size)

    return image


def dataset(data, shuffle=False, func=None, batch_size=1, repeat=None):
    """
    数据集处理，
    :param data: 数组 x_data
                 元组 (x_data, y_data,...)
                 字典 {"x":x_data, "y":y_data,...}
    :param shuffle: 是否进行shuffle
    :param func: 数据处理函数，输入输出格式与data相同
    :param batch_size: batch大小，默认为1
    :param repeat: 循环几次，默认无限循环
    :return: 迭代器
    """
    ds = ft.data_zip(data)

    if shuffle:
        np.random.shuffle(ds)

    if func is not None:
        ds = [i for i in map(func, ds)]

    stop = -1
    if repeat is not None:
        stop = repeat

    # 循环列表
    ring = Ring(ds)
    counter = ring.get_counter()
    batch_list = []

    # epochs循环，stop=-1条件恒成立，其余情况条件为0时退出循环
    while (stop - ring.get_counter()) != 0:
        # 清空list
        batch_list.clear()
        # 最后一次epoch并且剩余不足batch_size时，将batch_size的值设为剩余元素数量
        if ((stop - ring.get_counter()) == 1) and (
           (ring.length - ring.index) // batch_size == 0):
            batch_size = (ring.length - ring.index) % batch_size

        for _ in range(batch_size):
            # 每个epoch混洗一次
            if counter != ring.get_counter():
                np.random.shuffle(ds)
                counter = ring.get_counter()

            image_np = ring.next()
            image_np = np.array(image_np)

            image_np = ft.add_dim(image_np, 0)
            batch_list.append(image_np)

        yield ft.array_merge(batch_list, axis=0)
