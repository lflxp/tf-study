#!~/tf/venv/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) # 随机变量 比初始化为0好
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # out_size 表示多少列
    Wx_plus_b = tf.matmul(inputs,Weights) + biases  # 矩阵乘法+常量
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# input 层
x_data = np.linspace(-1,1,300)[:,np.newaxis] # 输入一个属性 
noise =  np.random.normal(0,0.05,x_data.shape) # 噪点
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])

ys = tf.placeholder(tf.float32,[None,1])

# 隐藏层
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1])) # 每个例子的平方求和

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # study_rate小于1 学习效率

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data)
    plt.ion() # show 后不暂停
    plt.show()
    for step in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if step % 50 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0]) # 图片中去除lines的第一个线段
            except Exception:
                pass

            predition_value = sess.run(predition,feed_dict={xs:x_data})
            lines = ax.plot(x_data,predition_value,'r-',lw=5)
            
            plt.pause(0.1)
