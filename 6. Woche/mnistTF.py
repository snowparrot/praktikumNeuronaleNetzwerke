## importieren
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets ("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #normalverteilung
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



with tf.name_scope('model'):
    x = tf.placeholder(tf.float32, [None, 784])
    
    W1 = weight_variable([784, 100])
    b1 = bias_variable([100])
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = weight_variable([100, 10])
    b2 = bias_variable([10])
    y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)
    
    y = y2

with tf.name_scope('train'):
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum (y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(600):
        batch_xs, batch_ys = mnist.train.next_batch(1000) #zufaellige Reihenfolge
        session.run(train_step, feed_dict= {x: batch_xs, y_: batch_ys })
    
    #Test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accurancy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
