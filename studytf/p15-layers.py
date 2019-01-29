#!/root/tf/venv/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) # 随机变量 比初始化为0好
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # out_size 表示多少列
    Wx_plus_b = tf.matmul(inputs,Weights) + biases  # 矩阵乘法+常量
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs