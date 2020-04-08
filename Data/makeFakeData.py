import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
# read in catchment characteristics file


currentDirectory = os.getcwd()
print(currentDirectory)

outputFileName = "fake_data"
if not os.path.exists(outputFileName):
    os.makedirs(outputFileName)
outputDir = outputFileName

numFakeData = 500000
numNewRecords = 0

for newDataIndex in range(numFakeData):
    if newDataIndex % 2 == 0:

        fakeX = np.zeros(950)
        fakeY = 0

    else:
        fakeX = np.ones(950)
        fakeY = 0

    fakeData = (fakeX,fakeY)

    
    fileName = "fake_01_" + str(newDataIndex)
    fakeDataPath = os.path.join(outputDir, fileName)
    pickle.dump(fakeData, open(fakeDataPath, "wb"))

    numNewRecords += 1


# make the indices for test, train, and validation
numNewRecords = numNewRecords - 1
numRecordsPath = os.path.join(outputDir, "numFiles")
pickle.dump(numNewRecords, open(numRecordsPath, "wb"))

indices = list(range(numNewRecords))
randomSeed = 42
np.random.seed(randomSeed)
np.random.shuffle(indices)

step = int(numNewRecords // 10)

print("numNewRecords: " + str(int(numNewRecords)))
print("step: " + str(step))


testIndices = indices[0:step]
valIndices = indices[step:int(2 * step)]
trainIndices = indices[int(2 * step):]

testIndicesPath = os.path.join(outputDir, "testIndices")
pickle.dump(testIndices, open(testIndicesPath, "wb"))

print("len testIndices: " + str(len(testIndices)))
print(testIndices[0:10])
valIndicesPath = os.path.join(outputDir, "validationIndices")
pickle.dump(valIndices, open(valIndicesPath, "wb"))

print("len valIndices: " + str(len(valIndices)))
print(valIndices[0:10])
trainIndicesPath = os.path.join(outputDir, "trainIndices")
pickle.dump(trainIndices, open(trainIndicesPath, "wb"))

print("len trainIndices: " + str(len(trainIndices)))
print(trainIndices[0:10])

