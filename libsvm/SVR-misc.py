#!/usr/bin/env python
from modshogun import RegressionLabels, RealFeatures
from modshogun import GaussianKernel, PolyKernel, CombinedKernel
from modshogun import MKLRegression, SVRLight
from modshogun import *
from numpy import *
import matplotlib.pyplot as plt
import xlrd


#################################################################################################


# Data preparation    
file_location = "/home/zhuozeying/Programs/python-program/test.xls"
workbook = xlrd.open_workbook(file_location)
#print 'data=', workbook
sheet1 = workbook.sheet_by_index(0)
#print 'sheet1=' , sheet1
rows = sheet1.nrows
print 'rows=' , rows
columns = sheet1.ncols
print 'columns=' , columns
# save the data into a list type
mydata = [[float(sheet1.cell_value(r,c)) for c in range(sheet1.ncols)] for r in range(sheet1.nrows)]
#print mydata
train_data = mydata[:336]
#print 'train_data', train_data
test_data = mydata[337:367]
#print 'test_data=', test_data
N_test = 30

#now transpose the data to take the right shape of shogun
#train_data = row_stack(train_data)
train_data = map(list, zip(*train_data))

test_data = row_stack(test_data)
test_data = map(list, zip(*test_data))


# now isolate the data from labels 
train_data_real = array(train_data[ : columns - 1], dtype = float64)
#print 'train_data_real=', train_data_real
train_data_labels = array(train_data[ columns - 1 : columns], dtype = float64)
#print 'train_data_labels=' , train_data_labels
# for c in train_data_labels:
# 	train_data_labels = c 
# 	train_data_labels = array(train_data_labels)
#print 'train_data_labels=' , train_data_labels

test_data_real = array(test_data[ : columns - 1], dtype = float64)
#print 'test_data_real=', test_data_real
test_data_labels = array(test_data[ columns - 1 : columns], dtype = float64)
# print 'test_data_labels=' , test_data_labels

# for c in test_data_labels:
# 	test_data_labels = c 
# 	test_data_labels = array(test_data_labels)
# print 'test_data_labels=' , test_data_labels

# print '#######################train_data_real######################################'
# print train_data_real
# print '#######################test_data_real######################################'
# print test_data_real
# print '#######################train_data_labels######################################'
# print train_data_labels
# print '#######################test_data_labels######################################'
# print test_data_labels


################################################################################################

labels=RegressionLabels(train_data_labels[0])
feats_train=RealFeatures(train_data_real)
feats_test=RealFeatures(test_data_real)

#################################################################################################
width = 80000
kernel = GaussianKernel(feats_train, feats_train, width)
svm_c = 100
svr_param = 1
svr_epsilon = LibSVR(svm_c, svr_param, kernel, labels, LIBSVR_EPSILON_SVR)
svr_epsilon.train()
svr_nu=LibSVR(svm_c, svr_param, kernel, labels, LIBSVR_NU_SVR)
svr_nu.train()
kernel.init(feats_train, feats_test)
out1_epsilon=svr_epsilon.apply().get_labels()
out2_epsilon=svr_epsilon.apply(feats_test).get_labels()
out1_nu=svr_epsilon.apply().get_labels()
out2_nu=svr_epsilon.apply(feats_test).get_labels()

Xlabel = []
for c in range(N_test):
	Xlabel.append(c + 1)
# print Xlabel
plt.plot(Xlabel, out1_epsilon, 'b')
plt.plot(Xlabel, test_data_labels[0], 'r')
plt.show()

plt.plot(Xlabel, out2_epsilon, 'g')
plt.plot(Xlabel, test_data_labels[0], 'r')
plt.show()

plt.plot(Xlabel, out1_nu, 'g')
plt.plot(Xlabel, test_data_labels[0], 'r')
plt.show()