# -*- coding: utf-8 -*-
"""
@FileName    : ring_test.py
@Description : None
@Author      : 齐鲁桐
@Email       : qilutong@yahoo.com
@Time        : 2019-05-17 11:08
@Modify      : None
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from fairy.classes import Ring


data = np.arange(30).reshape((5, 2, 3))

ring = Ring(data)

print(ring)

