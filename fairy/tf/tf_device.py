# -*- coding: utf-8 -*-
"""
@FileName    : tf_device.py
@Description : tf设备控制模块
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-16 22:41
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import tensorflow.keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)
