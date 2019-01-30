#!~/tf/venv/bin/python
# -*- coding: UTF-8 -*-
# https://www.bilibili.com/video/av16001891/?p=29
# 分类
import tensorflow as tf
import numpy as np

## Save to file
# remember to define the same drtype and shap when restore

# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')


# init = tf.initialize_all_variables()

# saver = tf.train.Saver()

# with tf.Session() as sess:
#         sess.run(init)
#         save_path = saver.save(sess,'my_net/save_net.ckpt')
#         print('Save to path',save_path)

# Restore variables
# redefine the same shape and same type for your variables
W_r = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights') # 6个数字 形状为2行3列
b_r = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases') 

# not need init step
saver = tf.train.Saver()

with tf.Session() as sess:
        saver.restore(sess, 'my_net/save_net.ckpt')
        print('weights:',sess.run(W_r))
        print('biases:',sess.run(b_r))