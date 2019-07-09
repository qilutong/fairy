# Fairy

> 作者：齐鲁桐  
> 邮件：qilutong@yahoo.com  

简单的应用实现  

安装  

使用setup.py安装

参数说明

或者直接将目录下的fairy/fairy这个文件夹copy到相关工程根目录下



对应文件说明

datasets    一些测试数据  
example     一些应用示例  
fairy       主程序  
test        开发测试程序，用户不需要关心


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