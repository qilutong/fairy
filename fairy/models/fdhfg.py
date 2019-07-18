# coding=UTF-8
import argparse
import os

import tensorflow as tf
from tensorflow.python.framework import graph_util

MODEL_DIR = "model/pb"
MODEL_NAME = "frozen_model.pb"

# 创建目录
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def freeze_graph(model_folder):
    # 检查目录下ckpt文件状态是否可用
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径

    output_node_names = "predictions"  # 原模型输出操作节点的名字
    # 得到图、clear_devices ：是否在导入期间清除“Operation”或“Tensor”的设备字段。
    saver = tf.train.import_meta_graph(
        input_checkpoint + '.meta', clear_devices=True)

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，
        # 不是操作节点的名字
        print("predictions : ",
              sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}))
        # 模型持久化，将变量值固定
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(
            output_graph_def.node))  # 得到当前图有几个操作节点

        for op in graph.get_operations():
            print(op.name, op.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_folder", type=str,
        help="input ckpt model dir")  # 命令行解析，help是提示符，type是输入的类型，
    # 这里运行程序时需要带上模型ckpt的路径，不然会报 error: too few arguments
    args = parser.parse_args()
    freeze_graph(args.model_folder)
    # freeze_graph("model/ckpt") #模型目录
