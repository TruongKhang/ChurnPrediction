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
    df = df.drop(['State', 'Phone_No'], axis=1)
    df['International_Plan'] = df['International_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Voice_Mail_Plan'] = df['Voice_Mail_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Customer_Left'] = df['Customer_Left'].map({True: 1, False: 0})
    dataTrain = np.array(df.values, dtype="float32")
    X_train = dataTrain[:, 0:18]
    Y_train = dataTrain[:, 18]

    df = pd.read_csv('churnTest.csv')
    df = df.drop(['State', 'Phone_No'], axis=1)
    df['International_Plan'] = df['International_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Voice_Mail_Plan'] = df['Voice_Mail_Plan'].str.strip().map({'yes': 1, 'no': 0})
    df['Customer_Left'] = df['Customer_Left'].map({True: 1, False: 0})
    dataTest = np.array(df.values, dtype="float32")
    X_test = dataTest[:, 0:18]
    Y_test = dataTest[:, 18]

    return X_train, Y_train, X_test, Y_test    
    
    
if __name__ == '__main__':
    X = np.random.randn(3,5)
    Y = np.random.randn(2,5)
    print X.T
    print Y.T
    minibatches = random_mini_batches(X, Y, 3, 3)