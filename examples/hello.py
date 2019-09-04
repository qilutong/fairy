# -*- coding: utf-8 -*-
"""
@FileName    : hello.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-08-19 21:18
@Modify      : None
"""
from __future__ import absolute_import, division, print_function



import glob
import fairy
import fairy.tail as ft
import tensorflow as tf

IMG_ROWS = None
IMG_COLS = None
CHANNELS = None



# 加载数据
image_names = ft.read_file_list("/warehourse/datasets/images/CelebA/Img/img_align_celeba")

image = fairy.data.load_image(image_names[0])


(IMG_ROWS, IMG_COLS, CHANNELS) = image.shape

print(IMG_ROWS, IMG_COLS, CHANNELS)











