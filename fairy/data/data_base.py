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
    打开图像文件
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
