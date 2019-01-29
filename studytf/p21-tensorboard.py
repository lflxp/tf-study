#!/root/tf/venv/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

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

# make up some real data 
x_data = np.linspace(-1,1,300)[:,np.newaxis] # 输入一个属性 
noise =  np.random.normal(0,0.05,x_data.shape) # 噪点
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name="x_input")
    ys = tf.placeholder(tf.float32,[None,1],name="y_input")

# 隐藏层
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,n_layer=2,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1])) # 每个例子的平方求和
    tf.summary.scalar('loss',loss) # 存量的变化

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # study_rate小于1 学习效率

# init = tf.initialize_all_variables()
# init = tf.global_variables_initializer()

sess = tf.Session()

# merged = tf.merge_all_summaries()
merged = tf.summary.merge_all()
# writer = tf.train.SummaryWriter('logs/',sess.graph)
writer = tf.summary.FileWriter('logs/',sess.graph)

# important step
sess.run(tf.global_variables_initializer())

for i in range (1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)