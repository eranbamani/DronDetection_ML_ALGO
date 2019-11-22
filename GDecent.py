#ERAN BAMANI
#17.12.16
#SGD funCTION
#===============================================

import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
#----------------------------------


class Gradient_Decent:
    def __init__(self, x, y, w, b, i, epoch, lr,theta, batch):
        self.iter = i
        self.epochs = epoch
        self.batch = batch
        self.y = y
        self.x = x
        self.b = b
        self.lr = lr
        self.w = w
        self.theta = theta

    def SGD(self):
        w = np.zeros([1,4])
        n = len(self.y).dtype('float')

        for t in range(self.epochs):
            p = w*self.x +self.b
            e = sum([data**2 for data in (self.y-p)])/n
            w_grad = -(2/n) * sum(self.x * (self.y - p))
            b_grad = -(2/n) * sum(self.y - p)

            w = w - (self.lr*w_grad)
            b = b - (self.lr * b_grad)
        return w, b, e

    def Calculate_cost(self, theta):
        m = len(self.y)
        pred = np.dot(self.x, theta)
        cost = (1/2 * m)*sum((pred-self.y)**2)
        return cost

    def Gradient_Decent(self):
        m = len(self.y)
        cost = np.zeros(self.iter)
        theta = np.zeros((self.iter, 2))
        for i in range(self.iter)
            pred = np.dot(self.x, self.w)
            theta = theta -(1/m)*self.lr*(self.x.T.dot((pred-self.y)))
            theta_array[i,:] = theta.T
            cost[i] = Calculate_cost(theta)

    def MiniBatch_grad(self, batchs, theta):

        m = self.y
        cost = np.zeros(self.iter)
        batch = int(m/batchs)

        for it in range(self.iter):
            c = 0
            ind = np.random.permutation(m)
            x = self.x[ind]
            y = self.y[ind]
            for itt in range(self.iter):
                xx = self.x[itt:itt+batch]
                yy = self.y[itt:itt + batch]
                xx = np.c_[np.ones(len(xx)), xx]
                pred = np.dot(xx, self.theta)

                theta = theta-(1/m)*self.lr*(xx.T.dot((pred-yy)))
                cost += self.Calculate_cost(theta, xx, yy)
            cost[it] = cost
        return theta, cost











