#!/root/tf/venv/bin/python
# -*- coding: UTF-8 -*-
# https://www.bilibili.com/video/av16001891/?p=22
# 分类
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') # 随机变量 比初始化为0好
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # out_size 表示多少列
            tf.summary.histogram(layer_name+'/biases',Weights)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases  # 矩阵乘法+常量
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
                tf.summary.histogram(layer_name+'/output',Weights)
            return outputs

# define placeholder for inputs to network

xs = tf.placeholder(tf.float32,[None,784]) # 28x28
ys = tf.placeholder(tf.float32,[None,784]) # 28x28
