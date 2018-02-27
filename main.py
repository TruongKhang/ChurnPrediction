# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 01:45:32 2018

@author: khangtg
"""

import tensorflow as tf
from utils import load_dataset
from model import neural_network, SVM, random_forest

def run():
    X_train, Y_train, X_test, Y_test = load_dataset()
    
    print ("SVM model...")
    SVM(X_train, Y_train, X_test, Y_test)
    print ("Done SVM")
    
    print ("Random forest model...")
    random_forest(X_train, Y_train, X_test, Y_test)
    print ("Done random forest")
    
    print ("Neural network model...")
    
    C = tf.constant(2, name='C')
    one_hot_matrix_train = tf.one_hot(Y_train, C, axis=0)
    one_hot_matrix_test = tf.one_hot(Y_test, C, axis=0)
    with tf.Session() as sess:
        one_hot_train = sess.run(one_hot_matrix_train)
        one_hot_test = sess.run(one_hot_matrix_test)
        
    Y_train = one_hot_train
    Y_test = one_hot_test
    X_train = X_train.T
    X_test = X_test.T
    
    weights = neural_network(X_train, Y_train, X_test, Y_test, [18,8,2], print_cost=True)
        
    return weights
    
if __name__ == '__main__':
    run()
    