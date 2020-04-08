import sys


inputFile = sys.argv[1]
outputFile = sys.argv[2]

source = 0
predicted = 1
target = 2

sourceData = []
predictedData = []
targetData = []


def getHeader(sourceCatchment, line):
    targetCatchment = line[2]
    yearOfBurn = line[3]
    startIndex = line[4]
    stopIndex = line[5]
    
    newHeader = [sourceCatchment, targetCatchment, yearOfBurn, startIndex, stopIndex]
    return newHeader

def getDiffRow(header, sourceData, targetData):
    
    assert (len(sourceData) == len(targetData)), "sourceData and targetData had different lengths!"
    
    diffs = []
    for j in range(len(sourceData)):
        sourceDay = sourceData[j]
        targetDay = targetData[j]

        if sourceDay != "None" and targetDay != "None":
            diff = str(sourceDay - targetDay)
        else:
            diff = "None"

        diffs.append(diff)

    return header + diffs

with open(outputFile, "w+") as outFile:
    outFile.write("sourceCatchment,targetCatchment, yearOfburn,startIndex(daysSince1972),stopIndex(daysSince1972),diffs->\n")


with open(inputFile, "r+") as inFile:
    
    # discard the header line
    inFile.readline()
    
    i = 0
    for line in inFile:
        
        line = line.replace('\n','')
        line = line.split(",")

        floatData = line[6:]
#        floatData = [float(a) for a in floatData]
        for index in range(len(floatData)):
            if floatData[index] != "None":
                floatData[index] = float(floatData[index])

        if i % 3 == source:
            sourceData = floatData
            sourceCatchment = line[2]
        elif i % 3 == predicted:
            predictedData = floatData
        elif i % 3 == target:
            targetData = floatData
            
            header = getHeader(sourceCatchment, line)
            diffRow = getDiffRow(header, sourceData, targetData)
            with open(outputFile, "a") as outFile:
                outFile.writelines(",".join(diffRow) + '\n')
                print(diffRow[0:20])
                print('new row')

        i = i + 1

