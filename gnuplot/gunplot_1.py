import os
import random
import math

os.chdir(u"c:\\mywork\\python\\gnuplot")

file = open("random_number.txt",'w+')

for i in range(200):
    file.write(str(i+random.random()))
    file.write(' ')
    file.write(str(math.log10(i+random.randint(-3,7))))
    file.write('\n')

file.close()