# -*- coding: utf-8 -*-
"""
@FileName    : convert_to_pb.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-04-01 19:25
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

model = 'model.pb'

output_graph_def = tf.GraphDef()


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


import numpy as np

np.random.randn()
tf.gfile.GFile(model, "rb")