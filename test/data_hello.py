# -*- coding: utf-8 -*-
"""
@FileName    : data_hello.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-07 13:43
@Modify      : None
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
from fairy.data import open_image
from fairy.data import save_image
from fairy.data.data_conversion import image_fc
import os

import fairy.tail as ft

raw_path = "../datasets/images/barbara_gray.bmp"

image_fc(raw_path, 'png')
image_fc(raw_path, 'png', new_path="2333")
image_fc(raw_path, 'png', mode="RGB")
image_fc(raw_path, 'png', new_path="23333", mode="GRAY")
