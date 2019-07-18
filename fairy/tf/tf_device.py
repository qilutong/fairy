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
import tensorflow.keras.backend as K


def init_config():
    """
    初始化Session
    :return:
    """
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)

    K.set_session(session)


def clear():
    """
    销毁当前的 TF 图并创建一个新图
    :return:
    """
    K.clear_session()


tf.get_default_session()
