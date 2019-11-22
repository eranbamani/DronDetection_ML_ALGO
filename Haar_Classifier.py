#ERAN BAMANI
#17.12.16
#Haar_Classifier funCTION
#===============================================
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
import integralImage_sum
# --------------------------
def Haar_type(case,ii, x,y,new_width,new_length, ttype):
    if case == 0:
        sum1 = integralImage_sum(ii, x, y, np.floor(new_width / 2), new_length)
        sum2 = integralImage_sum(ii, x + np.floor(new_width / 2), y, np.floor(new_width / 2), new_length)

    elif case == 1:
        sum1 = integralImage_sum(ii,x,y,new_width, np.floor(new_length/2))
        sum2 = integralImage_sum(ii,x,y + np.floor(new_length/2), new_width, np.floor(new_length/2))
        feature = (sum1-sum2)

    elif case == 2:
        [sum1] = integralImage_sum(ii, x, y, new_width, new_length)
        [sum2] = integralImage_sum(ii, x + np.ceil(new_width/2), y, new_width, new_length)
        [sum3] = integralImage_sum(ii, x + np.ceil(new_width/2), y, np.floor(new_width/2), new_length)
        feature = (sum1+sum2-3*sum3)

    else:
        [sum1] = integralImage_sum(ii, x, y,new_length, new_width)
        [sum2] = integralImage_sum(ii, x, y + np.ceil(new_length/2), new_width, new_length)
        [sum3] = integralImage_sum(ii, x, y + np.ceil(new_length/2), new_width, np.floor(new_length/2))
        feature = (sum1+sum2-3*sum3)

    return feature
























