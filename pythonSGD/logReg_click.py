from numpy import *  
import matplotlib.pyplot as plt  
import time  
  
  
# calculate the sigmoid function  
def sigmoid(inX):  
    return 1.0 / (1 + exp(-inX)) 
    
# train a logistic regression model using some optional optimize algorithm  
# input: train_x is a mat datatype, each row stands for one sample  
#        train_y is mat datatype too, each row is the corresponding label  
#        opts is optimize option include step and maximum number of iterations  
def trainLogRegres(train_x, train_y, opts):  
    # calculate training time  
    startTime = time.time()  
  
    numSamples, numFeatures = shape(train_x)  
    alpha = opts['alpha']; maxIter = opts['maxIter']  
    weights = ones((numFeatures, 1))  
  
    # optimize through gradient descent algorilthm  
    for k in range(maxIter):  
        if opts['optimizeType'] == 'gradDescent': # gradient descent algorilthm  
            output = sigmoid(train_x * weights)  
            error = train_y - output  
            weights = weights + alpha * train_x.transpose() * error  
        elif opts['optimizeType'] == 'stocGradDescent': # stochastic gradient descent  
            for i in range(numSamples):  
                output = sigmoid(train_x[i, :] * weights)  
                error = train_y[i, 0] - output  
                weights = weights + alpha * train_x[i, :].transpose() * error  
        elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth stochastic gradient descent  
            # randomly select samples to optimize for reducing cycle fluctuations   
            dataIndex = range(numSamples)  
            for i in range(numSamples):  
                alpha = 4.0 / (1.0 + k + i) + 0.01  
                randIndex = int(random.uniform(0, len(dataIndex)))  
                output = sigmoid(train_x[randIndex, :] * weights)  
                error = train_y[randIndex, 0] - output  
                weights = weights + alpha * train_x[randIndex, :].transpose() * error  
                del(dataIndex[randIndex]) # during one interation, delete the optimized sample  
        else:  
            raise NameError('Not support optimize method type!')  
      
  
    print ('Congratulations, training complete! Took %fs!' % (time.time() - startTime))  
    return weights  

# test your trained Logistic Regression model given test set  
def testLogRegres(weights, test_x, test_y):  
    numSamples, numFeatures = shape(test_x)  
    matchCount = 0  
    for i in xrange(numSamples):  
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5  
        if predict == bool(test_y[i, 0]):  
            matchCount += 1  
    accuracy = float(matchCount) / numSamples  
    return accuracy  
    
def loadData():  
    train_x = []  
    train_y = []  
    fileIn = open('train.csv')  
    for line in fileIn.readlines():  
        lineArr = line.strip().split()  
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])  
        train_y.append(float(lineArr[2]))  
    return mat(train_x), mat(train_y).transpose()  
  
###############################################################################  
## step 1: load data  
print ("step 1: load data...")  
train_x, train_y = loadData()  
test_x = train_x; test_y = train_y  
  
## step 2: training...  
print ("step 2: training...")  
opts = {'alpha': 0.01, 'maxIter': 2 ** 27, 'optimizeType': 'smoothStocGradDescent'}  
optimalWeights = trainLogRegres(train_x, train_y, opts)  
  
## step 3: testing  
print ("step 3: testing...")  
accuracy = testLogRegres(optimalWeights, test_x, test_y)  
  
## step 4: show the result  
print ("step 4: show the result...")    
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))    