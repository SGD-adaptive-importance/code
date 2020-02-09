import numpy as np
import random

###################
trainingFile = open("training.txt","r")
testFile = open("test.txt","r")
data = trainingFile.readlines()
testData = testFile.readlines()
Xmat = []
yvec = []
numData = len(data)
numTestData = len(testData)
testXmat = []
testyvec = []
#number of features in data
numFeatures = 123
#number of trials
numTrials = 5

#number of SGD steps
epochs = 21
numBatches = epochs*epochs
#step-size
stepSize = 1
regularizer = 0
###################


#Collect training data
for i in range(numData):
    words = data[i].split()
    yvec.append(int(words[0]))
    newRow = []
    numWords = len(words)-2
    for j in range(numWords):
        newRow = [0]*numFeatures
        nextWord = words[j+1]
        allSymbs = nextWord.split(":")
        newRow[int(allSymbs[0])]=1
    Xmat.append(newRow)

X = np.array(Xmat)
y = np.array(yvec)

#Collect test data
for i in range(numTestData):
    words = data[i].split()
    testyvec.append(int(words[0]))
    newRow = []
    numWords = len(words)-2
    for j in range(numWords):
        newRow = [0]*numFeatures
        nextWord = words[j+1]
        allSymbs = nextWord.split(":")
        newRow[int(allSymbs[0])]=1
    testXmat.append(newRow)


XBatches = []
yBatches = []

numRows = int(numData/numBatches)
rowCounter = 0

for i in range(numBatches):
    newXMatrix = []
    newyMatrix = []
    for j in range(numRows):
        newXMatrix.append(list(X[rowCounter]))
        newyMatrix.append(y[rowCounter])
        rowCounter+=1
    XBatches.append(newXMatrix)
    yBatches.append(newyMatrix)

#function that samples a gradient with probability distribution roughly that of importance sampling
#hinge-loss
def sampleScaledGradient(M, v, diff):
    #M is n by d, v is 1 by d, diff is 1 by n
    #Corresponds to X, w, and y
    outIndex = 0
    outGradient = 0
    maxNorm = 0
    #Put w in the correct format
    colV = (np.array([v])).T
    sumNorms = 0
    rowNum = 0
    for row in M:
        rowv = np.matmul((np.array([row])),colV)
        #check if max(0,1-y_i*x_i*w^T) >= 0
        if diff[rowNum]*rowv[0][0] >= 1:
            norm = 0
        else:
            #-y_i*x_i^T
            firstTerm = -1*diff[rowNum]*(np.array([row])).T/numRows
            #lambda*w
            secondTerm = regularizer*colV
            gradient = firstTerm+secondTerm
            norm = np.linalg.norm(gradient)
        sumNorms += norm
        trow = random.uniform(0,1)
        tnorm = norm/trow
        if tnorm > maxNorm:
            outIndex = rowNum
            maxNorm = tnorm
            scaling = sumNorms * numBatches / norm / numRows
            outGradient = scaling * gradient
        rowNum += 1
    return outGradient

#standard SGD with uniform sampling
def SGD(M, v, diff):
    i = random.randint(0,len(M)-1)
    colV = (np.array([v])).T
    row = M[i]
    rowv = np.matmul(np.array([row]),colV)
    if diff[i]*rowv[0][0] >= 1:
        outGradient = colV*0
    else:
        #-y_i*x_i^T
        firstTerm = -1*diff[i]*(np.array([row])).T/numRows
        #lambda*w
        secondTerm = regularizer*colV
        outGradient = firstTerm+secondTerm
    return outGradient

#update position with gradient and step-size
def update(pos,grad,step):
    npcolgrad = grad * step
    #npcolgrad is a column vector
    diff = (npcolgrad.T.tolist())[0]
    newPos = []
    for i in range(len(pos)):
        newPos.append(pos[i]-diff[i])
    return newPos

#objective on each timestep
def score(M,pos,diff):
    outScore = 0
    rowCounter = 0
    #Put w in the correct format
    colV = (np.array([pos])).T
    for row in M:
        rowv = np.matmul((np.array([row])),colV)
        rowScore = 1-diff[rowCounter]*rowv[0][0]
        if rowScore > 0:
            outScore += rowScore
    return outScore


wOneScores = [0]*epochs
wTwoScores = [0]*epochs

#collect objective values on test data
for j in range(numTrials):
    wOne=[]
    for i in range(numFeatures):
        wOne.append(random.uniform(0,10))
    wTwo = wOne
    for i in range(epochs):
        gradIS = sampleScaledGradient(XBatches[i],wOne,yBatches[i])
        gradSGD = SGD(XBatches[i],wTwo,yBatches[i])
        scoreWone = score(testXmat, wOne, testyvec)
        scoreWtwo = score(testXmat, wTwo, testyvec)
        #scoreY = np.linalg.norm(np.matmul(matA,y)-vecb)
        wOne = update(wOne, gradIS, stepSize)
        wTwo = update(wTwo, gradSGD, stepSize)
        #y = update(y, gradSGD, stepSize)
        wOneScores[i]=wOneScores[i]+scoreWone/numTrials
        wTwoScores[i]=wTwoScores[i]+scoreWtwo/numTrials

#formatting for plots
for i in range(epochs):
    print((i,int(wOneScores[i])))

for i in range(epochs):
    print((i,int(wTwoScores[i])))
