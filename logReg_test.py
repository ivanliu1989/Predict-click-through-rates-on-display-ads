
from numpy import *  
import matplotlib.pyplot as plt  
import time  
  
def loadData():  
    train_x = []  
    train_y = []  
    fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')  
    for line in fileIn.readlines():  
        lineArr = line.strip().split()  
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])  
        train_y.append(float(lineArr[2]))  
    return mat(train_x), mat(train_y).transpose()  
  
  
## step 1: load data  
print "step 1: load data..."  
train_x, train_y = loadData()  
test_x = train_x; test_y = train_y  
  
## step 2: training...  
print "step 2: training..."  
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}  
optimalWeights = trainLogRegres(train_x, train_y, opts)  
  
## step 3: testing  
print "step 3: testing..."  
accuracy = testLogRegres(optimalWeights, test_x, test_y)  
  
## step 4: show the result  
print "step 4: show the result..."    
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  
showLogRegres(optimalWeights, train_x, train_y)   