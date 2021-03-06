# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:18:14 2018

@author: khangtg
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from utils import random_mini_batches
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X, Y
    
def initialize_parameters(layers):
    W1 = tf.get_variable("W1", [layers[1], layers[0]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [layers[1], 1], dtype=tf.float32, initializer=tf.zeros_initializer)
    
    W2 = tf.get_variable("W2", [layers[2], layers[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [layers[2], 1], dtype=tf.float32, initializer=tf.zeros_initializer)
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters
    
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    
    return Z2
    
def compute_cost(Z2, Y):
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cost
    
def neural_network(X_train, Y_train, X_test, Y_test, layers, learning_rate=0.0001, num_epochs=1000, minibatch_size=32, print_cost=False):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    #costs = []
    
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(layers)
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0.
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            num_minibatches = len(minibatches)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches
                
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" %(epoch, epoch_cost))
                
        parameters = sess.run(parameters)
        
        correct_prediction = tf.equal(tf.argmax(Z, axis=0), tf.argmax(Y, axis=0))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        print("Parameters have been trained!")
        
    return parameters
    
def SVM(X_train, Y_train, X_test, Y_test):
    classifier = svm.SVC()
    classifier.fit(X_train, Y_train)
    
    Y_predicted = classifier.predict(X_test)
    correct_prediction = np.array(Y_test == Y_predicted, dtype=float)
    accuracy = np.mean(correct_prediction)
    
    print ("Test Accuracy: ", accuracy)
    
def random_forest(X_train, Y_train, X_test, Y_test):
    classifier = RandomForestClassifier(max_depth=10, random_state=0)
    classifier.fit(X_train, Y_train)
    
    print ("Features important!")
    print (classifier.feature_importances_)
    
    Y_predicted = classifier.predict(X_test)
    correct_prediction = np.array(Y_test == Y_predicted, dtype=float)
    accuracy = np.mean(correct_prediction)
    
    print ("Test Accuracy: ", accuracy)
    
    