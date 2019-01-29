#!/root/tf/venv/bin/python
import tensorflow as tf

print(tf.__version__)
# input1 = tf.placeholder(tf.float32,[2,2])
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)
output1 = tf.add(input1, input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1: [7.0],input2: [2.0]}))
    print(sess.run(output1,feed_dict={input1: [7.0],input2: [2.0]}))
