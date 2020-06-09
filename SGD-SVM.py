import numpy as np
import random
import time

###################
trainingFile = open("a9a_SVM.txt","r")
testFile = open("a9a_SVM_test.txt","r")
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
numTrials = 100

#number of SGD steps
epochs = 76
#numBatches = epochs*epochs
numBatches = 15*epochs
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
    words = testData[i].split()
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
            #next step computes -y_i*x_i^T
            firstTerm = -1*diff[rowNum]*(np.array([row])).T/numRows
            #next step computes lambda*w
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
        #next step computes -y_i*x_i^T
        firstTerm = -1*diff[i]*(np.array([row])).T/numRows
        #next step computes lambda*w
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
wThreeScores = [0]*epochs
wFourScores = [0]*epochs

wOneTimes = [0]*epochs
wTwoTimes = [0]*epochs
wThreeTimes = [0]*epochs
wFourTimes = [0]*epochs


computeTime = 0

#collect objective values on test data
for j in range(numTrials):
    wOne=[]
    for i in range(numFeatures):
        wOne.append(random.uniform(0,10))
    wTwo = wOne
    wThree = wOne
    wFour = wOne
    timeOne = 0
    timeTwo = 0
    timeThree = 0
    timeFour = 0
    computeStart = time.time()
    for i in range(epochs):
        #Uncomment to track scores
        #scoreWone = score(testXmat, wOne, testyvec)
        #scoreWtwo = score(testXmat, wTwo, testyvec)
        #scoreWthree = score(testXmat, wThree, testyvec)
        #scoreWfour = score(testXmat, wFour, testyvec)
        #wOneScores[i]=wOneScores[i]+scoreWone/numTrials
        #wTwoScores[i]=wTwoScores[i]+scoreWtwo/numTrials
        #wThreeScores[i]=wThreeScores[i]+scoreWthree/numTrials
        #wFourScores[i]=wFourScores[i]+scoreWfour/numTrials

        #Uncomment to track times
        wOneTimes[i]=wOneTimes[i]+timeOne/numTrials
        wTwoTimes[i]=wTwoTimes[i]+timeTwo/numTrials
        wThreeTimes[i]=wThreeTimes[i]+timeThree/numTrials
        wFourTimes[i]=wFourTimes[i]+timeFour/numTrials

        Xin = XBatches[i]
        Yin = yBatches[i]

        gradMixed = 0
        start = time.time()
        gradIS = sampleScaledGradient(Xin,wOne,Yin)
        wOne = update(wOne, gradIS, stepSize)
        stop = time.time()
        timeOne = timeOne + (stop - start)

        start = time.time()
        gradSGD = SGD(Xin,wTwo,Yin)
        wTwo = update(wTwo, gradSGD, stepSize)
        stop = time.time()
        timeTwo = timeTwo + (stop - start)

        start = time.time()
        gradMixedIS = sampleScaledGradient(Xin,wThree,Yin)
        gradMixedSGD = SGD(Xin,wThree,Yin)
        wThreeIS = update(wThree, gradMixedIS, stepSize)
        wThreeSGD = update(wThree, gradMixedSGD, stepSize)
        #Take the better of the two gradients for scoreThree
        scoreWthreeIS = score(Xin, wThreeIS, Yin)
        scoreWthreeSGD = score(Xin, wThreeSGD, Yin)
        if scoreWthreeIS < scoreWthreeSGD:
            wThree = wThreeIS
        else:
            wThree = wThreeSGD
        stop = time.time()
        timeThree = timeThree + (stop - start)

        start = time.time()
        if i < 25:
            gradFlip = sampleScaledGradient(Xin,wFour,Yin)
        else:
            gradFlip = SGD(Xin,wFour,Yin)
        wFour = update(wFour, gradFlip, stepSize)
        stop = time.time()
        timeFour = timeFour + (stop - start)
    computeEnd = time.time()
    computeTime = computeEnd-computeStart

#formatting for plots
#importance sampling
for i in range(epochs):
    if i%5 == 0:
        #uncomment to print scores
        #print((i,int(wOneScores[i])))
        #uncomment to print times
        print((i,round(wOneTimes[i],4)))
#SGD
for i in range(epochs):
    if i%5 == 0:
        #uncomment to print scores
        #print((i,int(wTwoScores[i])))
        #uncomment to print times
        print((i,round(wTwoTimes[i],4)))

#Mixed
for i in range(epochs):
    if i%5 == 0:
        #uncomment to print scores
        #print((i,int(wThreeScores[i])))
        #uncomment to print times
        print((i,round(wThreeTimes[i],4)))

#Flip
for i in range(epochs):
    if i%5 == 0:
        #uncomment to print scores
        #print((i,int(wFourScores[i])))
        #uncomment to print times
        print((i,round(wFourTimes[i],4)))

print(computeTime)
