#-*- coding: utf-8 -*-
import os
from svmutil import *
os.chdir('C:\mywork\ml\libsvm\python')
y, x = svm_read_problem('../heart_scale')

m = svm_train(y[:200], x[:200], '-c 4')

p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
#https://www.cnblogs.com/Dzhouqi/p/3653823.html