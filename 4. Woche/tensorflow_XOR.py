import numpy as np  
import pandas as pd  
import tensorflow as tf 
import matplotlib.pyplot as plt





d = {'x1' : [0., 1., 0., 1.],
     'x2' : [0., 0., 1., 1.],
     'y'  : [0., 1., 1., 0.]}
    
# reading xor data    
train = pd.DataFrame(d)
test = pd.DataFrame(d)

Xtrain = train.drop("y", axis=1)
Xtest =  train.drop("y", axis=1)

ytrain = train
ytest = test

def create_train_model(hidden_nodes, num_iters):
    # Reset the graph
    tf.reset_default_graph()
    
    X = tf.placeholder(shape=(4, 2), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(4, 3), dtype=tf.float64, name='y')
    
    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(2, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 1), dtype=tf.float64)
    
    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))
    
    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)
    
    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
        
# Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X:Xtrain.as_matrix(), y: ytrain.as_matrix()}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)        

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2
    

# Plot the loss function over iterations
num_hidden_nodes = [5, 10, 20]  
loss_plot = {5: [], 10: [], 20: []}  
weights1 = {5: None, 10: None, 20: None}  
weights2 = {5: None, 10: None, 20: None}  
num_iters = 2000

plt.figure(figsize=(12,8))  
for hidden_nodes in num_hidden_nodes:  
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)
    plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 2-%d-1" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)  
plt.ylabel('Loss', fontsize=12)  
plt.legend(fontsize=12)  
plt.show()





