# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir('c://mywork//ml//libsvm//python')
from svmutil import *

def svm_scale(y_s,x_s,limit_min,limit_max):
    x_min = min(x_s)  # x.min
    y_min = min(y_s)
    x_max = max(x_s)  # x.min
    y_max = max(y_s)
   # print x_min
   # print x_max
   # print y_min
   # print y_max
    x_scale = range(len(x_s))
    y_scale = range(len(x_s))
    for ii in range(len(x_s)):
        x_scale[ii] = limit_min + (limit_max - limit_min) * (x_s[ii]- x_min) / (x_max-x_min)
        y_scale[ii] = limit_min + (limit_max - limit_min) * (y_s[ii] - y_min) / (y_max - y_min)
      #  x1_s[ii] = float(x1_s[ii])
      #  y1_s[ii] = float(y1_s[ii])
    return (y_scale,x_scale,y_min,y_max,x_min,x_max)


def svm_restore(py,scale_min,scale_max,y_min,y_max,x_min,x_max):
    for ii in range(len(py)):
        py[ii] = y_min + (y_max - y_min) * (py[ii] - scale_min) / (scale_max - scale_min)
    return py

def array_to_svm(scale_y,scale_x):
    prob_y = []
    prob_x = []
    # x1 = {}

    #print scale_x
    for ii in range(len(scale_x)):
        x1 = {}
        x1[int(1)] = float(scale_x[ii])
        prob_y += [float(scale_y[ii])]
        prob_x += [x1]
    return (prob_y,prob_x)

def save_array_to_svmfile(y0,x0,filename):
    #output = open('c://mywork//python//libsvm//svrTest1.txt', 'w')
    output = open(filename,'w')
    for ii in range(len(x0)):
        txt_string = ""
        the_text = str(y0[ii]) + ' ' + str(int(1)) + ':' + str(x0[ii]) + '\n'
        output.write(the_text)
    output.close()

def array2_to_livsvm(y0,x0):
    prob_y = []
    prob_x = []
    for i, x_0 in enumerate(x0[:, 0]):
        a = {}
        a[int(1)] = float(x_0)
        a[int(2)] = float(x0[i, 1])
        prob_y += [float(y0[i])]
        prob_x += [a]
    return prob_y, prob_x