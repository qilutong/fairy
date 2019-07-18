# -*- coding: utf-8 -*-
"""
@FileName    : h5_to_pb.py
@Description : h5模型转换为pb格式
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-04-01 13:31
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import tensorflow.keras.backend as K

# 路径参数
input_path = 'input path'
weight_file = 'weight.h5'
weight_file_path = os.path.join(input_path, weight_file)
output_graph_name = weight_file[:-3] + '.pb'


# 转换函数
def h5_to_pb(h5_model,
             output_dir,
             model_name,
             out_prefix="output_",
             log_tensorboard=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util, graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(
        sess, init_graph, out_nodes)
    graph_io.write_graph(
        main_graph, output_dir, name=model_name, as_text=False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name), output_dir)


# 输出路径
output_dir = os.path.join(os.getcwd(), "trans_model")
# 加载模型
h5_model = tf.keras.models.load_model(weight_file_path)
h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
print('model saved')

