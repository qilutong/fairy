from setuptools import setup

import argparse

parser = argparse.ArgumentParser(description="安装工具包")

parser.add_argument("tool", help="需要安装的工具")
parser.add_argument("new_format", help="目标格式")

parser.add_argument("--tensorflow", "-tf", default=True, help="安装")

m_args = parser.parse_args()

packagges = [
    'fairy', 'fairy.pt', 'fairy.tf', 'fairy.tf2', 'fairy.data', 'fairy.tail',
    'fairy.models', 'fairy.classes', 'fairy.scripts'
]

setup(name='fairy',
      version='1.0.0',
      packages=[
          'fairy', 'fairy.tf', 'fairy.data', 'fairy.tail', 'fairy.models',
          'fairy.classes', 'fairy.scripts'
      ],
      url='https://github.com/qilutong/fairy',
      license='MIT',
      author='齐鲁桐',
      author_email='qilutong@yahoo.com',
      description='深度学习工具包')
