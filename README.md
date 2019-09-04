# Fairy

> 作者：齐鲁桐  
> 邮件：qilutong@yahoo.com  

简单的应用实现  

安装  

使用setup.py安装，默认安装tensorflow相关工具代码，需要依赖安装了tensorflow的环境

参数说明。。。。

或者直接将目录下的fairy/fairy这个文件夹copy到相关工程根目录下



对应文件说明

datasets    一些测试数据  
example     一些应用示例  
fairy       主程序  
test        测试程序


fairy相关包简介

| 名称     | 简介 |
| ---     | --- | 
| classes | 一些功能类 | 
| data    | 数据相关包 | 
| models  | 模型相关包 | 
| pt      | PyTorch 1.x封装 | 
| scripts | 脚本 | 
| tail    | 常用函数包 | 
| tf      | TensorFlow 1.x封装 | 
| tf2     | TensorFlow 2.0封装 | 


data
* data_generate 生成简单测试数据
* data_load     加载数据，转换为numpy数组
* data_save     保存数据

仿写tensorflow的DataSet，不依赖tensorflow，可单独在任意环境使用，用法类似tf.data.Dataset

例子：
```python
import tensorflow as tf
import fairy

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 将数据无限循环并进行shuffle，每个批次32个数据，最终得到一个生成器
data = fairy.data.Dataset(
    (x_train, y_train)).repeat().shuffle().batch(32).make_iterator()

model.fit_generator(data, epochs=5, steps_per_epoch=60000 // 32)
model.evaluate(x_test, y_test)
```

读取目录下下图像，并生成Numpy数组
```python
# 每次使用时读取，节省内存
images = fairy.data.load_images(
    "../datasets/images", mode="RGB", batch_size=2)

# # 将数据全部加载到内存中
# images = fairy.data.load_images_all(
#     "../datasets/images", mode="RGB", batch_size=2)
```
函数原型
```python
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
```

其余说明待续...