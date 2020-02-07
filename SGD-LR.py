import numpy as np
import random

###################
numTrials = 10
epochs = 40
numBatches = epochs*epochs
x=[]
for i in range(3072):
    x.append(random.uniform(0,10))
y = x
stepSize = 1*10**-13
###################

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='latin1')
    return batch

info = unpickle("data_batch_1")
A = list(info['data'])
b = list(info['labels'])

matA = np.array(A)
vecb = np.array(b)

test_info = unpickle("test_batch")
test_A = list(info['data'])
test_b = list(info['labels'])

test_matA = np.array(test_A)
test_vecb = np.array(test_b)


numRows = int(len(A)/numBatches)
ABatches = []
bBatches = []
rowCounter = 0
for i in range(numBatches):
    newAMatrix = []
    newbMatrix = []
    for j in range(numRows):
        newAMatrix.append(list(A[rowCounter]))
        newbMatrix.append(b[rowCounter])
        rowCounter+=1
    ABatches.append(newAMatrix)
    bBatches.append(newbMatrix)

def sampleScaledGradient(M, v, diff):
    #M is n by d, v is 1 by d, diff is 1 by n
    #Corresponds to A, x^T, and b
    outIndex = 0
    outGradient = 0
    maxNorm = 0
    #Put x in the correct format
    colV = (np.array([v])).T
    sumNorms = 0
    rowNum = 0
    for row in M:
        #b is d by 1
        #a_i^T*b, where a_i=d by 1
        rowv = np.matmul(np.array([row]),colV)
        #a_ia_i^T*b
        rowTrowv = np.matmul((np.array([row])).T,rowv)
        #2b^T*a_i
        twobTA = 2*diff[rowNum]*(np.array([row])).T
        gradient = rowTrowv - twobTA
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
    #outGradient is a column vector
    return outGradient

def update(pos,grad,step):
    npcolgrad = grad * step
    #npcolgrad is a column vector
    diff = (npcolgrad.T.tolist())[0]
    newPos = []
    for i in range(len(pos)):
        newPos.append(pos[i]-diff[i])
    return newPos

def SGD(M, v, diff):
    i = random.randint(0,len(M)-1)
    colV = (np.array([v])).T
    row = M[i]
    rowv = np.matmul(np.array([row]),colV)
    rowTrowv = np.matmul((np.array([row])).T,rowv)
    twobTA = 2*diff[i]*(np.array([row])).T
    outGradient = rowTrowv - twobTA
    return outGradient

allScores = []
for j in range(numTrials):
    trialData = []
    x=[]
    for i in range(3072):
        x.append(random.uniform(0,10))
    y = x
    for i in range(epochs):
        gradIS = sampleScaledGradient(ABatches[i],x,bBatches[i])
        gradSGD = SGD(ABatches[i],y,bBatches[i])
        scoreX = np.linalg.norm(np.matmul(test_matA,x)-test_vecb)
        scoreY = np.linalg.norm(np.matmul(test_matA,y)-test_vecb)
        x = update(x, gradIS, stepSize)
        y = update(y, gradSGD, stepSize)
        trialData.append((scoreX,scoreY))
    allScores.append(trialData)

summaryXScores = []
summaryYScores = []

for j in range(epochs):
    scoreX = 0
    scoreY = 0
    for i in range(numTrials):
        scoreX += allScores[i][j][0]
        scoreY += allScores[i][j][1]
    avgX = scoreX/numTrials
    avgY = scoreY/numTrials
    summaryXScores.append((j,int(avgX)))
    summaryYScores.append((j,int(avgY)))

for i in summaryXScores:
    print(i)

for j in summaryYScores:
    print(j)
