import numpy as np


rootFolder = "/home/sethbw/Documents/brian_flow_code/Data"

inFilePath = rootFolder + "/ref_burn_prediction_output_1.csv"
outFilePath = rootFolder + "/compressed_ref_burn_prediction.csv"

def stringToFloat(flowString):

    flowString = flowString.replace("\\n","")
    flowFloat = float(flowString)
    return flowFloat 

def compressPredictions(flowMatrix):
    
    maxLength = 0
    for j in range(len(flowMatrix)):
        if len(flowMatrix[j]) > maxLength:
            maxLength = len(flowMatrix[j])
    
    compression = []

    firstRow = flowMatrix[0]
    # go through every index
    for i in range(maxLength):
        iList = []
        # of every row
        for j in range(len(flowMatrix)):
            #print(flowMatrix[i])
            try:
                flowVal = flowMatrix[j][i]
                # FIXME: add test for null and convert to flaot, get rid of \n
                if flowVal != "":
                    flowVal = flowVal.replace("\\n","")
                    flowVal = float(flowVal)
                    iList.append(flowVal)
            except:
                pass
                # not all the rows are the same length

        # and average them
        mean = np.mean(iList)
        compression.append(mean)

    return compression


with open(outFilePath, "w+") as outFile:
    outFile.writelines("testNum,dataType,catchmentId,yearOfBurn,IndexOfStart(sr=1972),IndexOfStop(sr=1972)\n") # add the header




with open(inFilePath, "r+") as inFile:
    currentTest = 0
    flowPredictions = []

    target = 0
    source = 0
    
    i = 0
    for line in inFile:
        line = line.split(",")
        flow = line[6:-1]

#        print(line[1])
        if line[1] == "source":
            # save the compressed data
            if i > 1:
                compressedPrediction = compressPredictions(flowPredictions) 

                # write all to the outputFile 
                with open(outFilePath, "a+") as outFile:

                    source = [str(day) for day in source]
                    target = [str(day) for day in target]
                    compressedPrediction = [str(day) for day in compressedPrediction]

                    # add the source
                    sourceData = sourceMetadata + source
                    outFile.writelines(",".join(sourceData) + "\n")

                    # add the target
                    targetData = targetMetadata + target
                    outFile.writelines(",".join(targetData) + "\n")

                    # add the compressed values
                    predictedData = predictionMetadata + compressedPrediction
                    outFile.writelines(",".join(predictedData) + "\n")

                    print(len(target))
                    print(len(source))
                    print(len(flowPredictions))
                    print("**********")
            # now reset everything for the next prediciton
            flowPredictions = []
            sourceMetadata = line[0:6]
            source = flow
            sourceCatchment = line[2]

        elif line[1] == "target":
            target = flow
            targetMetadata = line[0:6]
            targetCatchment = line[2]

        elif line[1] == "predicted":
            predictionMetadata = line[0:6]
            flowPredictions.append(flow)

        i = i + 1



