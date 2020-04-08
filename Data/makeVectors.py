import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys

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
    sourceInfo = sourceCatchment + sourceFlow
    y = float(targetFlow[-1])
    targetFlow[-1] = 0 # zero the day we are predicting
    targetInfo = targetCatchment + targetFlow
    x = []
    x.append(sourceInfo)
    x.append(targetInfo)
    x = np.asarray(x, dtype=np.float32)

    return (x, y)

print('a')
# flowDataFile = "01_output_old_included.csv"
with open(flowDataFile, "r+") as inFile:
    numNewRecords = 0
    oldLine = None
    i = 0
    for line in inFile:
        print(i)
        if i == 1:
            oldLine = line
        if i > 1:
            # map from the old line to the new line
            newLine = line

            newData = newLine.split(",")
            oldData = oldLine.split(",")

            newCatchment = newData[0]
            oldCatchment = oldData[0]
            newFlow = newData[1:]
            oldFlow = oldData[1:]

            oldCatchmentCharacteristics = []
            newCatchmentCharacteristics = []

            # find the correct catchment characteristics
            with open(catchmentCharacteristicsFile, "r+") as characteristicsFile:
                j = 0
                for cLine in characteristicsFile:
                    if j > 0:
                        cLineList = cLine.split(",")
                        catchment = cLineList[0]
                        if int(catchment) == int(oldCatchment):
                            oldCatchmentCharacteristics = cLineList[1:]
                        if int(catchment) == int(newCatchment):
                            newCatchmentCharacteristics = cLineList[1:]
                    j = j + 1

            # check to make sure we found the catchment, and proceed
            if (len(oldCatchmentCharacteristics) == 0) or (len(newCatchmentCharacteristics) == 0):
                print("ERROR. didn't properly process characteristics data")
            else:
                # collect the next 14 days of data
                for dayIndex in range(len(newFlow) - 15): # stop before we run to the end of the file!
                    newDays = newFlow[dayIndex:dayIndex + 14] # these will both be part of the y and x
                    oldDays = oldFlow[dayIndex:dayIndex + 14] # these will both be part of the y and x

                    # go through each day and 1) make sure it is numeric, and 2) divide by 10,000

                    newDaysNumeric = []
                    oldDaysNumeric = []

                    for index in range(len(oldDays)):

                        newDay = newDays[index]
                        oldDay = oldDays[index]

                        if newDay.isnumeric():
                            newDaysNumeric.append(float(newDay) / 10000)
                        if oldDay.isnumeric():
                            oldDaysNumeric.append(float(oldDay) / 10000) # make the data less than 1 (or close to 1)

                    # if we can compare across the entire span of time
                    if len(oldDaysNumeric) == 14 and len(newDaysNumeric) == 14:

                        # make forward X and Y
                        data1 = makeXandY(oldCatchmentCharacteristics, oldDaysNumeric,
                                          newCatchmentCharacteristics, newDaysNumeric)
                        # make reverse X and Y
                        data2 = makeXandY(newCatchmentCharacteristics, newDaysNumeric,
                                          oldCatchmentCharacteristics, oldDaysNumeric)
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



