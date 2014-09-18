from numpy import *
dataMat = []; labelMat = []
fr = open('testSet.txt')
for line in fr.readlines():
    singleArr=[1.0]
    lineArr = line.strip().split()
    for i in range(2):
        singleArr.append(float(lineArr[i]))
    dataMat.append(singleArr)
    labelMat.append(int(lineArr[2]))

print ones(18)