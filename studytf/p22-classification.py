#!~/tf/venv/bin/python
# -*- coding: UTF-8 -*-
# https://www.bilibili.com/video/av16001891/?p=22
# https://www.bilibili.com/video/av9912938/?spm_id_from=333.788.videocard.4
# 分类
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') # 随机变量 比初始化为0好
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # out_size 表示多少列
        Wx_plus_b = tf.matmul(inputs,Weights) + biases  # 矩阵乘法+常量
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result # 输出百分比

# define placeholder for inputs to network

xs = tf.placeholder(tf.float32,[None,784]) # 28x28
ys = tf.placeholder(tf.float32,[None,10]) # 28x28

# add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax) # softmax做分类的激活函数

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) #loss 生成分类算法

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

# important step
sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch_xs,batch_ys = mnist.train.next_batch(100) # 将数据100个一次进行训练
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))