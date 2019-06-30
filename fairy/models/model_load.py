# -*- coding: utf-8 -*-
"""
@FileName    : model_load.py
@Description : 常见框架模型加载
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-04-09 16:36
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf


def load_graph(model_file):
    """
    从持久化模型加载到计算图
    :param model_file: 模型名称
    :return: 返回tensorflow计算图
    """
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph
