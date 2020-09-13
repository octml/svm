import os

os.chdir('C:\mywork\ml\libsvm\windows')
from svmutil import *

y, x = svm_read_problem('train.1')#read training data
yt, xt = svm_read_problem('test.1')#test
m = svm_train(y, x )#train
svm_predict(yt,xt,m)#test