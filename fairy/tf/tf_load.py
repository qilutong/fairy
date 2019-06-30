# -*- coding: utf-8 -*-
"""
@FileName    : tf_load.py
@Description : 加载tf模型
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-15 19:08
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf


def load_graph(model_file, name=None):
    """
    加载tf模型
    :param model_file: 模型文件名
    :param name: 节点名称
    :return: tf计算图
    """
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name=name)

    return graph

