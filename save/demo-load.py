#!~/tf/venv/bin/python
# -*- coding: UTF-8 -*-
# 将tensorflow导入为tf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
# 进口操作系统
# 将matplotlib.pyplot导入为plt
# 将numpy导入numpy
# 读取数据
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

# 加载模型
fig = plt.figure()
ax = fig.subplots(1,1)
with tf.Session(graph = tf.Graph()) as sess:
    tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.TRAINING],'./Model/')
    # sess.run(tf.global_variables_initializer()) # 加了这句话就会导致图初始化
    input_x = sess.graph.get_tensor_by_name('input/input_x:0')
    input_y = sess.graph.get_tensor_by_name('input/input_y:0')
    output = sess.graph.get_tensor_by_name('dense_1/BiasAdd:0')
    print(sess.run(tf.argmax(output,1),feed_dict={input_x:mnist.train.images[0:10,:]}))
    print(sess.run(tf.argmax(mnist.train.labels[0:10,:],1)))

# https://blog.csdn.net/tengxing007/article/details/56672556
# https://blog.csdn.net/rainmaple20186/article/details/80464178
# https://blog.csdn.net/lenbow/article/details/52181159