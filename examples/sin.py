# -*- coding: utf-8 -*-
"""
@FileName    : sin.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-08-28 22:01
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def gen_data():
    x = np.linspace(-np.pi, np.pi, 100)
    x = np.reshape(x, (len(x), 1))
    y = np.sin(x)
    return x, y


INPUT_NODE = 1
HIDDEN_NODE = 50
OUTPUT_NODE = 1
LEARNING_RATE = 0.001


def inference(input_tensor):
    with tf.name_scope('Layer-1'):
        weight = tf.Variable(tf.truncated_normal(
            shape=[INPUT_NODE, HIDDEN_NODE], stddev=0.1, dtype=tf.float32),
                             name='weight')
        bias = tf.Variable(
            tf.constant(0, dtype=tf.float32, shape=[HIDDEN_NODE]))
        l1 = tf.nn.relu(tf.nn.xw_plus_b(input_tensor, weight, bias))
    with tf.name_scope('Layer-2'):
        weight = tf.Variable(tf.truncated_normal(
            shape=[HIDDEN_NODE, OUTPUT_NODE], stddev=0.1, dtype=tf.float32),
                             name='weight')
        bias = tf.Variable(
            tf.constant(0, dtype=tf.float32, shape=[OUTPUT_NODE]))
        l2 = tf.nn.xw_plus_b(l1, weight, bias)
    return l2


def train():
    x = tf.placeholder(dtype=tf.float32,
                       shape=[None, INPUT_NODE],
                       name='x-input')
    y_ = tf.placeholder(dtype=tf.float32,
                        shape=[None, OUTPUT_NODE],
                        name='y-input')
    global_step = tf.Variable(0, trainable=False)
    logits = inference(x)
    loss = tf.reduce_mean(tf.square(y_ - logits))  # 均方误差
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        loss, global_step=global_step)
    train_x, train_y = gen_data()
    np.random.seed(200)
    shuffle_index = np.random.permutation(train_x.shape[0])  #
    shuffled_X = train_x[shuffle_index]
    shuffled_y = train_y[shuffle_index]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_x, train_y, lw=5, c='r')
    plt.ion()
    plt.show()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500000):
            feed_dic = {x: shuffled_X, y_: shuffled_y}
            _, train_loss = sess.run([train_step, loss], feed_dict=feed_dic)
            if (i + 1) % 80 == 0:
                print('loss at train data:  ', train_loss)
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                y_pre = sess.run(logits, feed_dict={x: train_x})
                lines = ax.plot(train_x, y_pre, c='black')
                plt.pause(0.1)


if __name__ == '__main__':
    train()
