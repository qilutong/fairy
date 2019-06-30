# -*- coding: utf-8 -*-
"""
@FileName    : data_conversion.py
@Description : 数据转换
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-07 11:13
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import os

import fairy.tail as ft

from .data_base import open_image


def image_fc(raw_path, new_format, new_path=None, mode=None, size=None):
    """
    将图像转换为另一格式
    :param raw_path: 图像初始路径,要包含文件名
    :param new_format: 要转换的格式
    :param new_path: 图像新路径，默认为 None
    :param mode: "GRAY" 或 "RPG"模式
    :param size: size
    :return:
    """
    # 判断读取模式RGB 或 GRAY
    image = open_image(raw_path, mode, size)
    # 判断新路径
    if new_path is None:
        image_name = os.path.basename(raw_path).split(".")[0] + "." + new_format
        image_name = os.path.join(os.path.dirname(raw_path), image_name)
    else:
        new_path = ft.check_sep(new_path)
        ft.check_dir(new_path)
        image_name = new_path + os.path.basename(raw_path).split(
            ".")[0] + "." + new_format
    print(image_name)
    image.save(image_name)
