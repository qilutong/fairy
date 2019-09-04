# -*- coding: utf-8 -*-
"""
@FileName    : tf_base.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-08-15 11:13
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

tf.enable_eager_execution()
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))
"""
超参数
"""
EPOCHS = 5
BATCH_SIZE = 32
"""
数据处理
"""
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(60000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)


class MyModel(Model):
    """
    构建模型类
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        """
        前向传播网络
        :param x: 输入数据
        :return:
        """
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = MyModel()  # 实例化模型

loss_object = tf.keras.losses.sparse_categorical_crossentropy  # loss函数

optimizer = tf.keras.optimizers.Adam()  # 优化器

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')  # 训练正确率

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')  # 验证正确率


def train_step(images, labels):
    """
    训练步骤
    :param images:
    :param labels:
    :return:
    """
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def test_step(images, labels):
    """
    测试步骤
    :param images:
    :param labels:
    :return:
    """
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, \
    Test Accuracy: {}'
    print(
        template.format(epoch + 1, train_loss.result(),
                        train_accuracy.result() * 100, test_loss.result(),
                        test_accuracy.result() * 100))
