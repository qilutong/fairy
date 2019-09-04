# -*- coding: utf-8 -*-
"""
@FileName    : dcgan.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-06 0:06
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

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model, Sequential

import fairy.tail as tf
import imageio

tf.enable_eager_execution()

BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 16
learning_rate = 1e-4


class DCGAN():
    """


    """

    def __init__(self):
        # 输入形状
        self.img_rows = None
        self.img_cols = None
        self.channels = None
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # 构建 generator
        self.generator = self.build_generator()

        # 构建并编译 discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 生成器将噪声作为输入并生成图像
        z = tf.keras.Input(shape=(self.latent_dim, ))
        img = self.generator(z)

        # 对于组合模型，我们只会训练 generator
        self.discriminator.trainable = False

        # 鉴定器将生成器生成的图像作为输入，并判断
        valid = self.discriminator(img)

        # 组合模型（合并生成器和鉴定器）
        # 训练生成器去欺骗鉴定器
        self.combined = tf.keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        """
        构建生成器
        :return: tf.keras.Model
        """
        model = tf.keras.Sequential()

        model.add(
            tf.keras.Dense(128 * 7 * 7,
                           activation="relu",
                           input_dim=self.latent_dim))
        model.add(tf.keras.Reshape((7, 7, 128)))
        model.add(tf.keras.UpSampling2D())
        model.add(tf.keras.Conv2D(128, kernel_size=3, padding="same"))
        model.add(tf.keras.BatchNormalization(momentum=0.8))
        model.add(tf.keras.Activation("relu"))
        model.add(tf.keras.UpSampling2D())
        model.add(tf.keras.Conv2D(64, kernel_size=3, padding="same"))
        model.add(tf.keras.BatchNormalization(momentum=0.8))
        model.add(tf.keras.Activation("relu"))
        model.add(tf.keras.Conv2D(self.channels, kernel_size=3,
                                  padding="same"))
        model.add(tf.keras.Activation("tanh"))

        model.summary()

        noise = tf.keras.Input(shape=(self.latent_dim, ))
        img = model(noise)

        return tf.keras.Model(noise, img)

    def build_discriminator(self):
        """
        构建鉴定器
        :return: tf.keras.Model
        """
        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.Conv2D(32,
                                   kernel_size=3,
                                   strides=2,
                                   input_shape=self.img_shape,
                                   padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(
            tf.keras.layers.Conv2D(64,
                                   kernel_size=3,
                                   strides=2,
                                   padding="same"))
        model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(
            tf.keras.layers.Conv2D(128,
                                   kernel_size=3,
                                   strides=2,
                                   padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(
            tf.keras.layers.Conv2D(256,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape)
        validity = model(img)

        return tf.keras.Model(img, validity)

    def generator_loss(self, generated_output):
        """
        生成器损失函数
        :param generated_output: 生成器网络输出
        :return:
        """
        return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output),
                                               generated_output)

    def discriminator_loss(real_output, generated_output):
        """
        鉴定器损失函数
        :param generated_output: 生成器网络输出
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

    def train(self, data, epochs, batch_size=128, save_interval=50):
        """
        进行训练
        :param epochs: 训练次数
        :param batch_size: 批次大小
        :param save_interval: 进行保存间隔
        :return:
        """

        # 加载数据
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


# 实例化
dcgan = DCGAN()

dcgan.train(epochs=14000, batch_size=256, save_interval=100)

