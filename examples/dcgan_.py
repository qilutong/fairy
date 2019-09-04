# -*- coding: utf-8 -*-
"""
@FileName    : dcgan_.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-08-18 23:37
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import fairy
import fairy.tail as ft
from IPython import display

import imageio

tf.enable_eager_execution()

BUFFER_SIZE = 60000
BATCH_SIZE = 64

EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 16

IMG_ROWS = None
IMG_COLS = None
CHANNELS = None

# 加载数据
image_names = ft.read_file_list(
    "/warehourse/datasets/images/CelebA/Img/img_align_celeba")

BUFFER_SIZE = len(image_names)

image = fairy.data.load_image(image_names[0])

(IMG_ROWS, IMG_COLS, CHANNELS) = image.shape

print(IMG_ROWS, IMG_COLS, CHANNELS)

import math

IMG_ROWS = math.ceil(IMG_ROWS / 8) * 8
IMG_COLS = math.ceil(IMG_COLS / 8) * 8


# 函数的功能是将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [IMG_ROWS, IMG_COLS])
    _image = (image_resized - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return tf.cast(_image, dtype=tf.float32)


# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices(image_names)

# shuffle在数据处理之前，混洗文件名而不是读取后的数组，buffer_size = 数据个数
dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

dataset = dataset.batch(BATCH_SIZE)


def make_generator_model():
    """

    :return:
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(IMG_ROWS // 8 * IMG_COLS // 8 * 256,
                              use_bias=False,
                              input_shape=(noise_dim, )))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((IMG_ROWS // 8, IMG_COLS // 8, 256)))

    print(model.output_shape)
    print(None, IMG_ROWS // 8, IMG_COLS // 8, 256)
    assert model.output_shape == (None, IMG_ROWS // 8, IMG_COLS // 8, 256)

    model.add(
        tf.keras.layers.Conv2DTranspose(128, (5, 5),
                                        strides=(1, 1),
                                        padding='same',
                                        use_bias=False))
    assert model.output_shape == (None, IMG_ROWS // 8, IMG_COLS // 8, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(64, (5, 5),
                                        strides=(2, 2),
                                        padding='same',
                                        use_bias=False))
    assert model.output_shape == (None, IMG_ROWS // 4, IMG_COLS // 4, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(32, (5, 5),
                                        strides=(2, 2),
                                        padding='same',
                                        use_bias=False))
    assert model.output_shape == (None, IMG_ROWS // 2, IMG_COLS // 2, 32)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(CHANNELS, (5, 5),
                                        strides=(2, 2),
                                        padding='same',
                                        use_bias=False,
                                        activation='tanh'))
    assert model.output_shape == (None, IMG_ROWS, IMG_COLS, CHANNELS)

    return model


def make_discriminator_model():
    """
    构建鉴定器
    :return: tf.keras.Sequential()
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def discriminator_loss(real_output, generated_output):
    """
    鉴定器损失函数
    :param real_output: 真实图片输出
    :param generated_output: 生成图片输出
    :return:
    """
    # [1,1,...,1] with real output since it is true and we want our
    # generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output),
        logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def generator_loss(generated_output):
    """
    生成器损失函数
    :param generated_output: 生成器网络输出
    :return:
    """
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output),
                                           generated_output)


generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

random_vector_for_generation = tf.random_normal(
    [num_examples_to_generate, noise_dim])


def train_step(images):
    """
    训练过程
    :param images:
    :return:
    """
    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # 真图片
        real_output = discriminator(images, training=True)
        # 假图片
        generated_output = discriminator(generated_images, training=True)
        # 生成模型损失
        gen_loss = generator_loss(generated_output)
        # 鉴定模型损失
        disc_loss = discriminator_loss(real_output, generated_output)
    # 求导
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                    discriminator.variables)
    # 更新参数
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.variables))

    print("one batch have done")


# 使用图计算
train_step = tf.contrib.eager.defun(train_step)


def generate_and_save_images(model, epoch, test_input):
    """

    :param model:
    :param epoch:
    :param test_input:
    :return:
    """

    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train(dataset, epochs):
    """

    :param dataset:
    :param epochs:
    :return:
    """
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            train_step(images)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1,
                                 random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(
            epoch + 1,
            time.time() - start))
    # generating after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, random_vector_for_generation)


train(dataset, EPOCHS)
