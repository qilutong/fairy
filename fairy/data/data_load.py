import numpy as np

import fairy.tail as ft

from .data_base import open_image
from ..classes import dataset


def load_image(image_name, mode=None, size=None):
    """读取图片并返回numpy数组"""
    # 判断读取模式RGB 或 GRAY
    image = open_image(image_name, mode, size)

    return np.array(image)


def load_images_all(path,
                    suffix=None,
                    size=(255, 255),
                    mode=None,
                    shuffle=False,
                    batch_size=None,
                    epochs=None):
    """
    读取目录下的图片并返回numpy数组生成器
    :param path: 图像路径
    :param suffix: 要匹配的图像后缀，可传入字符串、列表或元组
    :param size: 图像的长和宽，默认(255, 255)
    :param mode: 图像通道模式，默认保持原本格式，可替换为“RGB”和“GRAY”两种格式
    :param shuffle: 混洗，默认为False
    :param batch_size: 每一批次大小
    :param epochs: 总体循环次数，默认无限循环
    :return: numpy数组生成器，shape为（m * h * w * c）或（m * h * w）
    """
    # 判断要读取的图像后缀
    if suffix is None:
        suffix = ["png", "jpg", "jpeg", "svg", "bmp"]

    # 将目录下符合条件的图像名存入列表中
    image_names = ft.read_file_list(path, suffix)

    def open_img(image_name):

        image = load_image(image_name, mode=mode, size=size)

        return image

    ds = dataset.Dataset(image_names)
    ds = ds.batch(batch_size)
    ds = ds.shuffle(shuffle)
    ds = ds.map(open_img)
    ds = ds.repeat(epochs)

    return ds


def load_image_batch(image_names, mode, size):
    """
    按列表中的图像路径读取图像数据，并拼接位numpy数组
    :param image_names: 包含一批次图像名的列表
    :param mode: 图像读取模式
    :param size: 图像大小
    :return: numpy数组
    """
    images_list = []
    for image_name in image_names:
        image_np = load_image(image_name, mode=mode, size=size)
        image_np = image_np[np.newaxis, :]
        images_list.append(image_np)
    # 拼接数组
    return ft.array_merge(images_list, axis=0)


def load_images(path,
                suffix=None,
                size=(255, 255),
                mode=None,
                shuffle=False,
                batch_size=None,
                epochs=None):
    """
    读取目录下的图片并返回numpy数组生成器
    :param path: 图像路径
    :param suffix: 要匹配的图像后缀，可传入字符串、列表或元组
    :param size: 图像的长和宽，默认(255, 255)
    :param mode: 图像通道模式，默认保持原本格式，可替换为“RGB”和“GRAY”两种格式
    :param shuffle: 混洗，默认为False
    :param batch_size: 每一批次大小
    :param epochs: 总体循环次数，默认无限循环
    :return: numpy数组生成器，shape为（m * h * w * c）或（m * h * w）
    """
    # 判断要读取的图像后缀
    if suffix is None:
        suffix = ["png", "jpg", "jpeg", "svg", "bmp"]

    # 将目录下符合条件的图像名存入列表中
    image_names = ft.read_file_list(path, suffix)

    ds = dataset.Dataset(image_names)
    ds = ds.batch(batch_size)
    ds = ds.shuffle(shuffle)
    ds = ds.repeat(epochs)

    for data_batch in ds.make_iterator():
        yield load_image_batch(data_batch, mode, size)


def load_csv():
    pass


if __name__ == "__main__":
    print()
