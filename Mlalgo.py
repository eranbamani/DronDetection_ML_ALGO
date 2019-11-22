#ERAN BAMANI
#17.12.16
#Mlalgo_fun
#===============================================

import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
#----------------------------------

def Mlalgo(data):

    feature = data[:,1]
    label = data[:,2]

    assert(np.size(feature, 1) == np.size(label ,1))
    train = 0.7
    n, m = np.size(feature)
    rand_index = np.random.permutation(n)
    n_train = np.ceil(n * train)
    nn = n-n_train
    x_train = np.zeros((n_train, m))
    x_test = np.zeros((nn, m))
    y_train = np.zeros((n_train, 1))
    y_test = np.zeros((nn, 1))

    for i in range(n_train):

        x_train[i, :] = feature[rand_index(i),:]
        y_train[i] = label(rand_index[i])

    for j in range(n_train+1, n):

        x_test[j - n_train,:] = feature[rand_index[j],:]
        y_test[j - n_train] = label[rand_index[j]]



    assert(np.size(x_train, 1) + np.size(x_test, 1) == np.size(feature, 1));
    assert(np.size(y_train, 1) + np.size(y_test, 1) == np.size(feature, 1));







