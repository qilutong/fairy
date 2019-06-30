# -*- coding: utf-8 -*-
"""
@FileName    : image_change_format.py
@Description : 将一个目录下的图像批量转换为另一种格式（大小）
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-10 16:04
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import argparse
import logging

import fairy.tail as ft
from fairy.data.data_conversion import image_fc

LOG_FORMAT = ("%(asctime)s - %(filename)s[line:%(lineno)d] "
              "- %(levelname)s: %(message)s")
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# 参数设置
parser = argparse.ArgumentParser(description="批量修改图片格式")
parser.add_argument("path", help="原始文件目录")
parser.add_argument("new_format", help="目标格式")

parser.add_argument(
    "--format",
    "-F",
    default="png,jpg,jpeg,svg,bmp",
    help="要图像原始格式，用,隔开：jep,png,bmp")
parser.add_argument(
    "--mode", "-M", choices=["RGB", "GRAY"], default=None, help="图像模式")
parser.add_argument("--size", "-S", default=None, help="size")
parser.add_argument("--new_path", "-N", default=None, help="保存到新路径")
parser.add_argument(
    "--version", "-V", action="version", version="%(prog)s 1.0", help="版本号")


def main(main_args):
    logging.info("开始")
    logging.info("path is {}".format(main_args.path))
    logging.info("format is {}".format(main_args.format))
    image_names = ft.read_file_list(main_args.path,
                                    main_args.format.split(","))

    new_format = main_args.new_format
    new_path = main_args.new_path
    mode = main_args.mode
    size = list(map(int, main_args.size.split(",")))
    for path in image_names:
        logging.info("file is {}".format(path))
        image_fc(path, new_format, new_path=new_path, mode=mode, size=size)

    logging.info("结束")


if __name__ == '__main__':
    m_args = parser.parse_args()
    main(m_args)
