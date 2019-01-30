#!~/tf/venv/bin/python
# -*- coding: UTF-8 -*-
#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

# 加载Graph
def loadGraph(dir):
    f = tf.gfile.FastGFile(dir,'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_graph =tf.import_graph_def(graph_def,name='')
    return persisted_graph

graph = loadGraph('graph/soft.ph')


with tf.Session(graph=graph) as sess:
    #sess.run(tf.initialize_all_variables())
    #sess.run(init) #加载时候不需要进行初始化
    softmax_tensor = sess.graph.get_tensor_by_name('layer/final_result:0')
    x = sess.graph.get_tensor_by_name('input/x_input:0')
    y_ = sess.graph.get_tensor_by_name('input/y_input:0')
    # name = sess.graph.get_tensor_by_name('tengxing:0')
    Weights = sess.graph.get_tensor_by_name('layer/W/Weights:0')
    biases = sess.graph.get_tensor_by_name('layer/b/biases:0')

    #W = tf.Variable(tf.zeros([784, 10]), name='Weights')
    #b = tf.Variable(tf.zeros([10]), name='biases')
    # tf.add_to_collection(tf.GraphKeys.VARIABLES, name)
    tf.add_to_collection(tf.GraphKeys.VARIABLES, Weights)
    tf.add_to_collection(tf.GraphKeys.VARIABLES, biases)
    try:
        saver = tf.train.Saver(tf.global_variables())  # 'Saver' misnomer! Better: Persister!
    except:
        pass
    print("load data")
    #print sess.run(name) 此时才有一个Tensor获取变量还要进行赋值
    saver.restore(sess, "data/soft.ckpt")  # now OK creted by tengxing
    #test
    correct_prediction = tf.equal(tf.argmax(softmax_tensor, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
# --------------------- 
# 作者：tengxing007 
# 来源：CSDN 
# 原文：https://blog.csdn.net/tengxing007/article/details/56672556 
# 版权声明：本文为博主原创文章，转载请附上博文链接！

# https://blog.csdn.net/huachao1001/article/details/78501928
# 导入训练好的模型
# 在第1小节中我们介绍过，tensorflow将图和变量数据分开保存为不同的文件。因此，在导入模型时，也要分为2步：构造网络图和加载参数

# 3.1 构造网络图
# 一个比较笨的方法是，手敲代码，实现跟模型一模一样的图结构。其实，我们既然已经保存了图，那就没必要在去手写一次图结构代码。

# saver=tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
# 1
# 上面一行代码，就把图加载进来了

# 3.2 加载参数
# 仅仅有图并没有用，更重要的是，我们需要前面训练好的模型参数（即weights、biases等），本文第2节提到过，变量值需要依赖于Session，因此在加载参数时，先要构造好Session：

# import tensorflow as tf
# with tf.Session() as sess:
#   new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
#   new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
# 1
# 2
# 3
# 4
# 此时，W1和W2加载进了图，并且可以被访问：

# import tensorflow as tf
# with tf.Session() as sess:    
#     saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
#     print(sess.run('w1:0'))
# ##Model has been restored. Above statement will print the saved value
# 1
# 2
# 3
# 4
# 5
# 6
# 执行后，打印如下：

# [ 0.51480412 -0.56989086]
# --------------------- 
# 作者：huachao1001 
# 来源：CSDN 
# 原文：https://blog.csdn.net/huachao1001/article/details/78501928 
# 版权声明：本文为博主原创文章，转载请附上博文链接！