# -*- coding: utf-8 -*-
"""
@FileName    : tf_ops.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-07-14 上午4:28
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.InteractiveSession()


def tf_init():
    tf.global_variables_initializer()


def session_init():
    """
    初始化session，返回一个默认的session
    :return:
    """
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    return tf.InteractiveSession(config=config)
