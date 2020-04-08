import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys
import random

# arguments:
# python3.7 makeVectors.py characteristicsFile, outputFolderName, flowDataFile

catchmentCharacteristicsFile = sys.argv[1]
outputFolderName = sys.argv[2]
flowDataFile = sys.argv[3]
filePrefix = sys.argv[4]

# read in the arguments

# read in catchment characteristics file

# read in catchment flow data file
# catchmentCharacteristicsFile = "normalized_zerod_catchment_characteristics.csv"

# three vectors from different catchments concattenated on each other
currentDirectory = os.getcwd()
print(currentDirectory)

# outputFolderName = "01_data"
if not os.path.exists(outputFolderName):
    os.makedirs(outputFolderName)
outputDir = outputFolderName

def makeXandY(sourceCatchment, sourceFlow, targetCatchment, targetFlow):
    sourceInfo = (float(sourceCatchment), sourceFlow)
    y = float(targetFlow[-1])
    targetFlow[-1] = 0 # zero the day we are predicting
    targetInfo = (float(targetCatchment), targetFlow)
    x = []
    x.append(sourceInfo)
    x.append(targetInfo)
#    x = np.asarray(x, dtype=np.float32)

    return (x, y)

print('a')
# flowDataFile = "01_output_old_included.csv"
with open(flowDataFile, "r+") as inFile:
    isTest = True
    isVal = False
    isTrain = False

    numNewRecords = 0
    oldLine = None
    i = 0
    for line in inFile:
        print(i)
        if i == 1:
            oldLine = line
        if i > 1:
            ranNum = random.randint(0,10)
            if randNum == 0:
                isTest = True
                isVal = False
                isTrain = False
            elif randNum == 1:
                isTest = False
                isVal = True
                isTrain = False
            else:
                isTest = False
                isVal = False
                isTrain = True

            # map from the old line to the new line
            newLine = line

            newData = newLine.split(",")
            oldData = oldLine.split(",")

            newCatchment = newData[0]
            oldCatchment = oldData[0]
            newFlow = newData[1:]
            oldFlow = oldData[1:]

            # collect the next 14 days of data
            for dayIndex in range(len(newFlow) - 15): # stop before we run to the end of the file!
                newDays = newFlow[dayIndex:dayIndex + 14] # these will both be part of the y and x
                oldDays = oldFlow[dayIndex:dayIndex + 14] # these will both be part of the y and x

                # go through each day and make sure it is numeric

                newDaysNumeric = []
                oldDaysNumeric = []
                
                for index in range(len(oldDays)):

                    newDay = newDays[index]
                    oldDay = oldDays[index]

                    if newDay.isnumeric():
                        newDaysNumeric.append(int(newDay))
                    if oldDay.isnumeric():
                        oldDaysNumeric.append(int(oldDay)) # make the data less than 1 (or close to 1)

                # if we can compare across the entire span of time
                if len(oldDaysNumeric) == 14 and len(newDaysNumeric) == 14:

                    # make forward X and Y
                    data1 = makeXandY(oldCatchment, oldDaysNumeric,
                                        newCatchment, newDaysNumeric)
                    # make reverse X and Y
                    data2 = makeXandY(newCatchment, newDaysNumeric,
                                        oldCatchment, oldDaysNumeric)
                    # save them
                    newFileName1 = filePrefix + str(numNewRecords)
                    newFilePath1 = os.path.join(outputDir, newFileName1)
                    pickle.dump(data1, open(newFilePath1, "wb"))
                    numNewRecords += 1

                    newFileName2 = filePrefix + str(numNewRecords)
                    newFilePath2 = os.path.join(outputDir, newFileName2)
                    pickle.dump(data2, open(newFilePath2, "wb"))
                    numNewRecords += 1

            print(numNewRecords)

            # move forward 1 line
            oldLine = line

        i = i + 1

    # make the indices for test, train, and validation
    numNewRecords = numNewRecords - 1
    numRecordsPath = os.path.join(outputDir, "numFiles")
    pickle.dump(numNewRecords, open(numRecordsPath, "wb"))

    indices = list(range(numNewRecords))
    randomSeed = 42
    np.random.seed(randomSeed)
    np.random.shuffle(indices)

    step = int(numNewRecords // 10)

    testIndices = indices[0:step]
    valIndices = indices[step:int(2 * step)]
    trainIndices = indices[int(2 * step):]

    testIndicesPath = os.path.join(outputDir, "testIndices")
    pickle.dump(testIndices, open(testIndicesPath, "wb"))

    valIndicesPath = os.path.join(outputDir, "validationIndices")
    pickle.dump(valIndices, open(valIndicesPath, "wb"))

    trainIndicesPath = os.path.join(outputDir, "trainIndices")
    pickle.dump(trainIndices, open(trainIndicesPath, "wb"))



