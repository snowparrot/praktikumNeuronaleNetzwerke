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
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



with tf.name_scope('modelc'):
    X = tf.placeholder(tf.float32, [None, 784])
    
    #layer conv 1
    
    x_image = tf.reshape(X, [-1, 28, 28, 1])
    
    W_conv1 = weight_variable([5, 5, 1, 32]) # 5 x 5 Bilder: 32 Neuronen gucken auf die Bilder
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    
    #layer pool1
    h_pool1 = max_pool_2x2(h_conv1)
    # layer conv 2
    
    W_conv2 = weight_variable([5, 5, 32, 64]) # 32 Features wegen oben (5. 5. 32 Eingaben)  64 Features
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
    #layer Pool2
    h_pool2 = max_pool_2x2(h_conv2)
    
    # Dense Layer
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 7 x 7 Größé des Bilder 64 Features 
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #drop out
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout (h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    # softmax
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    
with tf.name_scope('train'):
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
with tf.name_scope('eval'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    
    for i in range(2000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = session.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
            print(i, train_accuracy)
            
        session.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.4})
    print(session.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))
    
