"""
fairy 顶级目录
"""
from __future__ import absolute_import, division, print_function

import os
import sys

from . import classes
from . import data
from . import models
from . import pt
from . import scripts
from . import tail
from . import tf
from . import tf2

# 添加顶级包路径到环境变量
env_path = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.dirname(env_path)
sys.path.append(env_path)
del env_path

# 导入限制
__all__ = ["classes", "data", "models", "pt", "scripts", "tail", "tf", "tf2"]
