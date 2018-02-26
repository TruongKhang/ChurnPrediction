# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:54:32 2018

@author: khangtg
"""

import numpy as np
import pandas as pd

def random_mini_batches(X, Y, minibatch_size, seed):
    X = X.T
    Y = Y.T
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)
    
    m = len(X)
    minibatches = []
    
    num_minibatches = int(m / minibatch_size)
    
    for i in range(num_minibatches):
        minibatch_X = X[(i*minibatch_size) : ((i+1)*minibatch_size)]
        minibatch_Y = Y[(i*minibatch_size) : ((i+1)*minibatch_size)]
        minibatches.append((minibatch_X.T, minibatch_Y.T))
        
    if (num_minibatches * minibatch_size < m):
        minibatch_X = X[num_minibatches*minibatch_size : m]
        minibatch_Y = Y[num_minibatches*minibatch_size : m]
        minibatches.append((minibatch_X.T, minibatch_Y.T))
    
    return minibatches
    
def load_dataset():
    df = pd.read_csv('churnTrain.csv')
    df = df.drop(columns=['State', 'Phone_No'])
    df['International_Plan'] = df['International_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Voice_Mail_Plan'] = df['Voice_Mail_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Customer_Left'] = df['Customer_Left'].map({True: 1, False: 0})
    dataTrain = df.values
    X_train = dataTrain[:, 0:19]
    Y_train = dataTrain[:, 19]

    df = pd.read_csv('churnTest.csv')
    df = df.drop(columns=['State', 'Phone_No'])
    df['International_Plan'] = df['International_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Voice_Mail_Plan'] = df['Voice_Mail_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Customer_Left'] = df['Customer_Left'].map({True: 1, False: 0})
    dataTest = df.values
    X_test = dataTest[:, 0:19]
    Y_test = dataTest[:, 19]

    return X_train.T, Y_train.T, X_test.T, Y_test.T    
    
    
if __name__ == '__main__':
    X = np.random.randn(3,5)
    Y = np.random.randn(2,5)
    print X.T
    print Y.T
    minibatches = random_mini_batches(X, Y, 3, 3)