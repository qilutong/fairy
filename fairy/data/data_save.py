# -*- coding: utf-8 -*-
"""
@FileName    : data_save.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-05 15:49
@Modify      : None
"""
from __future__ import absolute_import, division, print_function
from PIL import Image


def save_image(data, path):
    """将numpy数组保存为图像格式"""
    img = Image.fromarray(data)
    img.save(path)

