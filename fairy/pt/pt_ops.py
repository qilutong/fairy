# -*- coding: utf-8 -*-
"""
@FileName    : pt_ops.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-23 15:00
@Modify      : None
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import torch

# 检测是否支持GPU
gpu = torch.cuda.is_available()


def get_device():
    """
    获取可用设备
    :return: 字符串 ‘CPU’ 或 ‘GPU’
    """
    return torch.device('cuda:0' if gpu else 'cpu')


# 设备
device = get_device()


def info():
    """PyTorch 信息"""
    print("PyTorch version: ", torch.__version__)
    if gpu:
        print("CUDA version: ", torch.version.cuda)
        print("cuDNN version: ", torch.backends.cudnn.version())
        print("GPU: ", torch.cuda.get_device_name())


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    if gpu:
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def clear_gpu():
    """
    清除GPU缓存
    :return:
    """
    torch.cuda.empty_cache()


