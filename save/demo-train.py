#!~/tf/venv/bin/python
# 将tensorflow导入为tf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 将numpy导入numpy
# 将matplotlib.pyplot导入为plt

#整个网络结构:卷积 - >池化 - >卷积 - >池化 - >全连接 - > dropout->全连接 - > SOFTMAX

#MINST数据
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
#
#define占位符
def tf.name_scope('input'): #输入采用了name_scope的形式,也可以不采用,直接从图上得到
    xs = tf.placeholder(tf.float32,[None,28 * 28],name ='input_x')#输入图像shape:[？,784]
    ys = tf.placeholder(tf.float32,[None,10],name ='input_y')#标签
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs,[-1,28,28,1])

#卷积图层1
conv1 = tf.layers.conv2d(
    输入= x_image,
    filters = 16,#输出空间的维数
    kernel_size =(5,5),#二维卷积窗口的高度和宽度strides =(
    1,1), #沿高度和宽度的卷积步长
    padding ='same',#相同或有效
    activation = tf .nn.relu 
)#[ -  1,28,28,16] #max 
# 池 
pool1 = tf.layers.max_pooling2d(
    inputs = conv1,
    pool_size =(2,2),#pooling操作strides的
    步幅=(2,2),#池化操作的步幅
    padding ='same'#same或valid 
)#[ -  1,14,14,16]#卷积 
二层 
conv2 = tf.layers.conv2d(pool1,32,5,(1,1),'same',activation = tf.nn.relu)#[ -  1,14,14,32] #max 
轮询
POOL2 = tf.layers.max_pooling2d(CONV2,(2,2),(2,2),'相同')#[ -  1,7,7,32] 
#平坦
扁平= tf.reshape(POOL2,[ -  1,7 * 7 * 32]) 
#完全连接的层1 
dense1 = tf.layers.dense(
    输入=平,#张量输入
    单元= 256 ,则输出空间的维数#
    活化= tf.nn.relu 
)
#漏失
dense_drop = tf.layers.dropout(
    inputs = dense1,
    rate = keep_prob 
)
#预测
#与tf.name_scope( '输出'):#可以通过name_scope来确定变量域,但是博主是直接根据图来看的
预测= tf.layers.dense(dense_drop,10)#输出结果
#损失
损失= tf.losses.softmax_cross_entropy(
    onehot_labels = YS,
    logits =预测
)
#火车
train_op = tf.train.AdamOptimizer(0.005).minimize(loss)

# 准确性
accury= tf.metrics.accuracy(#创建两个局部变量
    labels = tf.argmax(ys,1),
    预测= tf.argmax(预测,1)
)[1]

成本= []#损失信息
sess = tf.Session()
"""
  local_variables_initializer用于计算精度
"""
sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
writer = tf.summary.FileWriter('./ logs /',sess.graph)#构建TensorBoard的Graph,不作详细介绍

#训练
for in range(1001):
    train_images,train_labels = mnist.train.next_batch(100)
    _,loss_out = sess.run([train_op,loss],{xs:train_images,ys:train_labels,keep_prob:0.5})
     #测试精确度
    if i%50 == 0:
        cost.append(loss_out)
        test_imges,test_label = mnist.test.next_batch(100)
        print(i,sess.run(accury,{xs:test_imges,ys:test_label}))

print(sess.run(tf.argmax(prediction,1),{xs:mnist.train.images [0:10,:]}))
print(sess.run(tf.argmax(mnist.train.labels [0:10,:],1)))

# #训练完成,保存训练好的网络
builder = tf.saved_model.builder.SavedModelBuilder('./ Models /')#保存前需要保证这个文件夹为空或者不存在
builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.TRAINING])
builder.save()