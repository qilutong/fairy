# -*- coding: utf-8 -*-
"""
@FileName    : tf_save.py
@Description : 保存模型
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-15 18:05
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util


def save_ckpt(sess, model_name):
    """
    模型保存为ckpt格式
    :param sess: 要保存的计算图会话
    :param model_name: 模型名
    :return:
    """
    saver = tf.train.Saver()  # 声明saver用于保存模型
    saver.save(sess, model_name)  # 模型保存


def save_bp(sess, model_name, node):
    """
    保存为bp文件格式
    :param sess: 要保存的计算图会话
    :param model_name: 模型名
    :param node: 要保存的节点名，列表格式
    :return:
    """
    # 得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    # 模型持久化，将变量值固定
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        node  # 需要保存的节点
    )
    with tf.gfile.GFile(model_name, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
