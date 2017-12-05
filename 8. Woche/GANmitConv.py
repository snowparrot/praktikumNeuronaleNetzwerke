#Import the libraries we will need.
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
        
        


# The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
# They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image

    return img
    
    



def generator(z):
    zP = slim.fully_connected(z, 4 * 4 * 256, normalizer_fn=slim.batch_norm, \
                              activation_fn=tf.nn.relu, scope='g_project', weights_initializer=initializer)
    zCon = tf.reshape(zP, [-1, 4, 4, 256])

    gen1 = slim.convolution2d_transpose( \
        zCon, num_outputs=64, kernel_size=[5, 5], stride=[2, 2], \
        padding="SAME", normalizer_fn=slim.batch_norm, \
        activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer)

    gen2 = slim.convolution2d_transpose( \
        gen1, num_outputs=32, kernel_size=[5, 5], stride=[2, 2], \
        padding="SAME", normalizer_fn=slim.batch_norm, \
        activation_fn=tf.nn.relu, scope='g_conv2', weights_initializer=initializer)

    gen3 = slim.convolution2d_transpose( \
        gen2, num_outputs=16, kernel_size=[5, 5], stride=[2, 2], \
        padding="SAME", normalizer_fn=slim.batch_norm, \
        activation_fn=tf.nn.relu, scope='g_conv3', weights_initializer=initializer)

    g_out = slim.convolution2d_transpose( \
        gen3, num_outputs=1, kernel_size=[32, 32], padding="SAME", \
        biases_initializer=None, activation_fn=tf.nn.tanh, \
        scope='g_out', weights_initializer=initializer)

    return g_out


def discriminator(bottom, reuse=False):
    x = tf.placeholder(tf.float32, [None, 784])
    
    #layer conv 1
    
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    
    gen1 = slim.convolution2d( \
        x_image, num_outputs=32, kernel_size=[5, 5], stride=[1, 1], \
        padding="SAME", normalizer_fn=slim.batch_norm, \
        activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer)    
    
    
    #W_conv1 = weight_variable([5, 5, 1, 32]) # 5 x 5 Bilder: 32 Neuronen gucken auf die Bilder
    #b_conv1 = bias_variable([32])
    
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    
    #layer pool1
    h_pool1 = max_pool_2x2(gen1)
    
    #h_pool1 = slim.max_pool2d(gen1, kernel_size = [1, 2, 2, 1], stride = [1, 2, 2, 1], padding='SAME')
    
    
    
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
    
    #max
    
    max_sig = tf.reduce_max(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    # sigmoid
    
    y_conv = tf.nn.sigmoid(max_sig)

tf.reset_default_graph()

z_size = 100 #Size of z vector used for generator.

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#These two placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #Random vector
real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32) #Real images

Gz = generator(z_in) #Generates images from random z vectors
Dx = discriminator(real_in) #Produces probabilities for real images
Dg = discriminator(Gz,reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

tvars = tf.trainable_variables()

#The below code is responsible for applying gradient descent to update the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #Only update the weights for the discriminator network.
g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #Only update the weights for the generator network.

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)

batch_size = 128 #Size of image batch to apply at each iteration.
iterations = 500000 #Total number of iterations to use.
sample_directory = './figs' #Directory to save sample images from generator in.
model_directory = './models' #Directory to save trained model to.

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(iterations):
        zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
        xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
        xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
        xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
        _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs}) #Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs}) #Update the generator, twice for good measure.
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs})
        if i % 10 == 0:
            print("Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss))
            z2 = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate another z batch
            newZ = sess.run(Gz,feed_dict={z_in:z2}) #Use new z to get sample images from generator.
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            #Save sample generator images for viewing training progress.
            save_images(np.reshape(newZ[0:36],[36,32,32]),[6,6],sample_directory+'/fig'+str(i)+'.png')
        if i % 1000 == 0 and i != 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
            print("Saved Model")


sample_directory = './figs'  # Directory to save sample images from generator in.
model_directory = './models'  # Directory to load trained model from.
batch_size_sample = 36

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # Reload the model.
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_directory)
    saver.restore(sess, ckpt.model_checkpoint_path)

    zs = np.random.uniform(-1.0, 1.0, size=[batch_size_sample, z_size]).astype(np.float32)  # Generate a random z batch
    newZ = sess.run(Gz, feed_dict={z_in: z2})  # Use new z to get sample images from generator.
    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)
    save_images(np.reshape(newZ[0:batch_size_sample], [36, 32, 32]), [6, 6],
                sample_directory + '/fig' + str(i) + '.png')
