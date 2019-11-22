#ERAN BAMANI
#17.12.16
#drone_algortihm_fun
#===============================================

import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
#----------------------------------

class Drone_Algortihm_Fun:

    def __init__(self, len_face, total, label, width, lenn, counter,features):

        self.len_face = len_face
        self.total = total
        self.label = label
        self.width = width
        self.lenn = lenn

        self.err = 0.7
        self.data = np.zeros([total, 2])
        self.tests = 28
        self.count = 0
        self.counter = counter
        self.features = features



    def Adaboots(self):

        for i in range(self.lenn-np.floor(self.lenn/5)):
            for j in range(self.width-np.floor(self.width/5)):
                while self.count < self.tests:
                    self.count += 1
                    Haar_type = divmod(self.count, len(self.label))
                    new_width = np.random.rand(np.floor(self.lenn/5))
                    new_length = np.random.rand(np.floor(self.width/5))

                    for ii in range(self.len_face):
                        iii = integralImage(:,:,ii)
                        feature =  Haar_Classifier(iii, i, j, new_width, new_length, Haar_type)
                        self.data[ii,1] = feature
                        self.data[ii,2] = -1

                    w, min_c = Mlalgo(self.data)
                    if min_c<0.3*self.total*self.err:
                        feature[1, self.count] = i
                        feature[2, self.count] = j
                        feature[3, self.count] = new_width
                        feature[4, self.count] = new_length
                        feature[5, self.count] = w

    def cascade(self):
        rate = 0.8
        error = 0
        cascade = 0
        ep = np.random.rand(self.total)
        for final in range(ep):

            i = np.random.rand(self.total)
            for s in range(self.counter):

                Haar_type = divmod(s, len(self.label))
                ii = integralImage(:,:, i)
                feature = Haar_Classifier(ii, self.features(1, s), self.features(2, s), self.features(3, s), self.features(4, s), Haar_type)
                if np.sign(self.features * self.features(5, s)) > 0:

                    cascade = cascade + 1

            if cascade >= self.counter * rate:
                error = error + 1
            else:
                error = error - 1

        return error

































