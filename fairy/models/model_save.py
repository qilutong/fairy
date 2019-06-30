# -*- coding: utf-8 -*-
"""
@FileName    : model_save.py
@Description : 常见框架模型保存
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-04-09 16:16
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util


def tf_save(sess, model_name, node=None):
    """
    tensorflow模型保存
    :param sess: 要保存的tensorflow会话
    :param model_name: 模型名称，包含路径
    :param node: 要保存的节点名
    :return:
    """
    if model_name.split(".")[-1] == "ckpt":
        saver = tf.train.Saver()
        saver.save(sess, model_name)

    elif model_name.split(".")[-1] == "pb":
        if node is None:
            raise ValueError("need a node name")
        graph_def = tf.get_default_graph().as_graph_def()
        # 模型持久化，将变量值固定
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            node  # 需要保存节点的名字
        )
        with tf.gfile.GFile(model_name, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
