#-*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir('c://mywork//ml//libsvm//python')
from svmutil import *
from mylibsvm import *

#training_file  = "dataset_23_train.txt"
#testing_file   = "dataset_23_test.txt"
def file_process():
    os.chdir('c://mywork//python//holy')
    os.system("svm-scale -y -1 1 -s scale dataset_23_train.txt> holy_train.scale")
    os.system("svm-scale -r scale dataset_23_test.txt > holy_test.scale")
    os.chdir('c://mywork//ml//libsvm//python')
    #os.system("python easy.py c://mywork//python//holy//holy_train.scale c://mywork//python//holy//holy_test.scale")
    #os.system("python grid.py c://mywork//python//holy//holy_train.scale")
    #os.system("svm - train holy_train.scale")
    #os.system("svm-predict holy_test.scale holy_train.scale.model holy_test.predict")

def test1():
    y, x = svm_read_problem('c://mywork//python//holy//holy_train.scale')#读入训练数据
    yt, xt = svm_read_problem('c://mywork//python//holy//holy_test.scale')#训练测试数据
    #param = svm_parameter('-s 1 -t 2 -c 1000 ')  # Guass
    model = svm_train(y, x )#训练
    p_label, p_acc, p_val = svm_predict(yt, xt, model)  #测试
    print yt
    print p_label
    #print p_acc
    #print p_val

def test2():
    from sklearn.datasets import make_blobs
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
#file_process()
test1()
