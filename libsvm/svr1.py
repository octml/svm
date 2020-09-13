# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mylibsvm import *
os.chdir('c://mywork//ml//libsvm//python')
from svmutil import *

""""
    x0 = np.sort(5 * np.random.rand(40, 1), axis=0)
    y0 = np.sin(x0).ravel()
    y0[::5] += 3 * (0.5 - np.random.rand(8)) # Add noise to targets
    #y, x = [1,-1], [{1:1, 2:1}, {1:-1,2:-1}]
#x1 = [{1:-1},{2:-0.9},{3:-0.8},{4:-0.7},{5:-0.6},{6:-0.5}]
#x1 = [[-1],[-0.9],[-0.8],[-0.7],[-0.6],[-0.5]]
#x = np.arange(-1,1,0.1)
#x = list(x1)
#y = list(y1)
  y2 = [0.09]
    x2 = [{1:0.3}]
    py,mse,val = svm_predict([1],x2,model)
    print py,mse,val
    #py,mse,p = svm_predict(prob_y,prob_x,model)

"""



def svr_function():

    x = np.linspace(-1, 1, 21)
    # x0 = np.arange(-1,1,0.1)
    y = map(lambda x: x ** 2, x)

    #x = np.sort(5 * np.random.rand(40))  # as colume
    #y = np.sin(x)
    x = np.sort(5 * np.random.rand(40, 1), axis=0) #  40 * 1 as colume
    y = np.sin(x).ravel()
    x = x.ravel()
    y[::5] += 3 * (0.5 - np.random.rand(8)) # Add noise to targets
    #print x.T
    print y
    return (y,x)



if __name__ == '__main__':    # x0 = np.linspace(-1,1,21)
    y0,x0 = svr_function()

    save_array_to_svmfile(y0,x0,'c://mywork//python//libsvm//svrTest1.txt')
    #svm-scale -y -1 1 -s scale svrTest1.txt > svrTest1Scale.txt
    #python gridregression.py -log2c -10,10,1 -log2g -10,10,1 -log2p -10,10,1 -v 10 -s 3 -t 2 svrTest1Sclae.txt > svrTest1Train.txt
    scale_min = -1;
    scale_max = 1;
    scale_y,scale_x,y_min,y_max,x_min,x_max = svm_scale(y0,x0,scale_min,scale_max)
    prob_y,prob_x = array_to_svm(scale_y,scale_x)
    #prob_y, prob_x = svm_read_problem('c://mywork//python//libsvm//svrTest1Scale.txt')
    #print prob_x
    #print prob_y
    prob = svm_problem(prob_y, prob_x)
    # 建模回归模型
    param = svm_parameter('-s 3 -t 2 -c 2.2 -g 2.8 -h 0 -p 0.01')  # Guass
    # param = svm_parameter('-s 3 -t 2 -c 1e3 -g 0.1 -p 0.01') # Guass
    # param = svm_parameter('-s 3 -t 1 -c 1e3') # linear

    model = svm_train(prob, param)
    # 利用建立的模型看其在训练集合上的回归效果
    py, mse, p = svm_predict(prob_y, prob_x, model)
    py = svm_restore(py, scale_min, scale_max, y_min, y_max, x_min, x_max)
    #print py #
    #print mse
    #print mse
    plt.plot(x0, y0, 'o');
    plt.plot(x0, py, '-');
    # plt.plot(x0,py,'-');
    plt.show()

# 进行预测
# testx = 1.1;
# display('真实数据')
# testy = -testx.^2

# [ptesty,tmse] = svmpredict(testy,testx,model);
# display('预测数据');

# 原文地址：https://sites.google.com/site/kittipat/libsvm_matlab
