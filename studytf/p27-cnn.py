#!~/tf/venv/bin/python
# -*- coding: UTF-8 -*-
# https://www.bilibili.com/video/av16001891/?p=22
# https://www.bilibili.com/video/av9912938/?spm_id_from=333.788.videocard.4
# 分类
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result # 输出百分比

# 设置权重
def weight_variable(shape):
    inital  = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

# 设置标量
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积核在W的最后一位定义的
def conv2d(x,W): # x是整个图片的信息
    # stride [1,x_movement,y_movement,1]
    # strides 步长 
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') 

# 池化
def max_pool_2x2(x):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784]) # 28x28
ys = tf.placeholder(tf.float32,[None,10]) # 28x28
keep_prob = tf.placeholder(tf.float32)
# 第一个值 -1代表了batch ，意思是由多少个图片， 里面含义是有n个28*28， 最后一个数字是（channel,）通道为1的图片， n根据传入的参数自己匹配
x_image = tf.reshape(xs,[-1,28,28,1])
# print(x_image.shape) # [n_samples,28,28,1]

## conv1 layer ##
# 5,5 patch 5x5
# 1 in size, 1个单位 长宽
# 32 out size 32个单位高度 想象成长方体的高
# 32和64都是卷积核的数量 即高度就算核的数量
# 卷积核的数量应该就算输出高度除以输入高度
# 第一层是因为输入高度是1 所以卷积核的数量是32
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # 非线性处理 output size 28x28x32
h_pool1 = max_pool_2x2((h_conv1)) # 整个层的输出值 output size 14x14x32
## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5,in size 32,out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) # 非线性处理 output size 14x14x32
h_pool2 = max_pool_2x2((h_conv2)) # 整个层的输出值 output size 7x7x32
## func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 变平 
#  [n_samples,7,7,64] => [n_smaples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# 处理过拟合 dropout
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2) # softmax 算概论 分类
# prediction = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2) # softmax 算概论 分类

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) #loss 生成分类算法

# 1e-4 = 0.0004
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

# important step
sess.run(tf.global_variables_initializer())

for i in range(200):
    batch_xs,batch_ys = mnist.train.next_batch(100) # 将数据100个一次进行训练
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5}) # 防止过拟合只在train过程中生效,1为不随机丢掉数据
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

# Save
saver = tf.train.Saver()

save_path = saver.save(sess,'my_net/save_net.ckpt')
print('Save to path',save_path)