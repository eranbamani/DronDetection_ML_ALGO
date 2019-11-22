#ERAN BAMANI
#17.12.16
#integral_vector_w function
#===============================================
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
from GDecent import *
# --------------------------

def integralImage_sum(ii,x,y,new_width,new_length):
    A = ii[x, y]
    B = ii[x + new_width, y]
    C = ii[x, y + new_length]
    D = ii[x + new_width, y + new_length]
    sum = D + A - B - C
    return sum

def w_vector(x_train,x_test,y_train,y_test):
    c_vec = 2**(np.linspace(-5,2,15))
    EP = 5
    err_vec = np.zeros((1, len(c_vec)))
    n, m = np.size(x_train)
    N, M = np.size(x_test)
    train_norm = np.zeros((n, m))
    test_norm = np.zeros((N, M))
    max_train = np.zeros((1, m))
    max_test = np.zeros((1, M))
    min_train = np.zeros((1, m))
    min_test = np.zeros((1, M))

    for i in range(m):

        max_train[i] = max(x_train[:, i])
        min_train[i] = min(x_train[:, i])
        train_norm[:, i] = (x_train[:, i] - min_train[i]) / (max_train[i] - min_train[i])

    for j in range(M):

        max_test[j] = max(x_test[:, j])
        min_test[j] = min(x_test[:, j])
        test_norm[:, j] = (x_test[:, j] - min_test[i])/(max_test[i] - min_test[i])

    for q in range(len(c_vec)):

        temp = c_vec[q]
        err_avg = 0
        for ii in range(EP):

            w, b, e = SGD(train_norm, y_train, temp)



        err_vec[q] = e / EP

    min_c = c_vec(err_vec == min(err_vec))
    w = SGD(x_train, y_train, min_c)
    w = np.mean(w)
    return w, min_c
