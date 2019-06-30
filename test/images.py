# -*- coding: utf-8 -*-
"""
@FileName    : images.py
@Description : 测试图像读取 pipeline
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-20 10:13
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import fairy

# 每次使用时读取，节省内存
images = fairy.data.load_images(
    "../datasets/images", mode="RGB", batch_size=2)

# # 将数据全部加载到内存中
# images = fairy.data.load_images_all(
#     "../datasets/images", mode="RGB", batch_size=2)

for img in images:
    print(img.shape)
