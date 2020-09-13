#-*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir('c://mywork//ml//libsvm//python')
from svmutil import *
from mylibsvm import *
from sklearn.datasets import make_blobs
def test1():
    y, x = svm_read_problem('c://mywork//python//libsvm//train.1')#读入训练数据
    yt, xt = svm_read_problem('c://mywork//python//libsvm//test.1')#训练测试数据
    model = svm_train(y, x )#训练
    p_label, p_acc, p_val = svm_predict(yt, xt, model)  #测试
    print yt
    print p_label
    #print p_acc
    #print p_val

def test2():
# we create 40 separable points
    x0, y0 = make_blobs(n_samples=40, centers=2, random_state=6)
    #plt.scatter(x0[:, 0], x0[:, 1], c=y0, s=30, cmap=plt.cm.Paired)
    #plt.show()
    prob_y,prob_x = array2_to_livsvm(y0,x0)
    print prob_y
  #  print prob_x
    prob = svm_problem(prob_y, prob_x)
# 建模回归模型
    param = svm_parameter('-s 0 -t 0 -c 1000 ')  # Guass
    model = svm_train(prob, param)
# 利用建立的模型看其在训练集合上的回归效果
    py, mse, p = svm_predict(prob_y, prob_x, model)
    print model

def test3():
    train_file = 'c://mywork//python//libsvm//train.1.scale.txt'
    test_file = 'c://mywork//python//libsvm//test.1.scale.txt'

    # svm-scale -y -1 1 -s scale test.1 > test.1.scale.txt
    # svm-scale -y -1 1 -s scale train.1 > train.1.scale.txt
    # rate, param = find_parameters(train_file, '-log2c -3,3,1 -log2g -3,3,1')
    y, x = svm_read_problem(train_file)  # 读入训练数据
    yt, xt = svm_read_problem(test_file)  # 训练测试数据

test3()
