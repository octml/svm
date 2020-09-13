# -*- coding: utf-8 -*-
# code:myhaspl@qq.com
# 7-17.py

#import matplotlib.pyplot as plt
import numpy as np
import os
y = [3, 5, 8, 5, 12, 26, 20]
#ind = np.array([1,2,3,4,5,6,7])
x = [1, 2, 3, 3, 6, 12, 11]

#output = open('svmdata.txt', 'w')
data = np.array([y[0],x[0]])
for ii in xrange(1,len(x)):
    ti = np.array([y[ii],x[ii]])
    data = np.row_stack((data,ti))
np.savetxt('svmdata.txt',data)


# read data file
readin = open('svmdata.txt', 'r')
# write data file
output = open('libsvm.txt', 'w')
index = 1
try:
    the_line = readin.readline()
    #print the_line
    while the_line:
        # delete the \n
        the_line = the_line.strip('\n')
      #  index = 0;
        output_line = ''
        for sub_line in the_line.split('\t'):
            if sub_line != 'NULL':
                txt_string = sub_line.split(' ')
                #print txt_string
                the_text = txt_string[0]+ ' ' + str(index) + ':' + txt_string[1]
                #print the_text
                output_line = output_line + the_text
            index = index + 1
        output_line = output_line + '\n'
       # print output_line
        output.write(output_line)
        the_line = readin.readline()
finally:
    readin.close()
#a = np.loadtxt('svmdtat.txt')