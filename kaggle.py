from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('train.csv')
    for line in fr.readlines():
    	singleArr=[1.0]
        lineArr = line.strip().split()
        for i in range(39):
        	singleArr.append(float(lineArr[i+2]))
        dataMat.append(singleArr)
        labelMat.append(int(lineArr[1]))
    return dataMat,labelMat

