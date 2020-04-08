

referenceCatchments = []
referenceCatchmentsPath = "/home/sethbw/Documents/brian_flow_code/Data/reference_watersheds.csv"
with open(referenceCatchmentsPath, "r+") as flowFile:
    i = 0
    for line in flowFile:
        if i > 0:
            lineList = line.split(",")
            referenceCatchment = lineList[0]
            if len(referenceCatchment) == 7:
                catchment = "0" + referenceCatchment
            referenceCatchments.append(referenceCatchment)
        i = i + 1



burnFilePath = "catchments_for_Q_analysis_corrected.csv"

burnedCatchments = []
with open(burnFilePath) as burnFile:
    i = 0
    for line in burnFile:
        if i > 0:
            data = line.split(",")
            catchment = data[0]
                    
            if len(catchment) == 7:
                catchment = "0" + catchment

        burnedCatchments.append(catchment)

def getColumns(outFileName, burn):

    with open("dailyFlowColumns.csv") as columnFile:
        # open the first column, get the indices of 
        i = 0
        for line in columnFile:
            lineList = line.split(",")

            if i == 0: # get the header
                j = 0
                keeperIndices = [j]
                for element in lineList:
                    if element in referenceCatchments:

                        if element in burnedCatchments:
                            if burn:
                                keeperIndices.append(j)
                        else:
                            if not burn:
                                keeperIndices.append(j)
    
                    j = j + 1

                with open(outFileName, "w+") as outFile:
                    keeperList = []
                    for index in range(len(lineList)):
                        if index in keeperIndices:
                            keeperList.append(lineList[index])
                    outFile.write(",".join(keeperList) + "\n")
        
            else:
                with open(outFileName, "a+") as outFile:
                    keeperList = []
                    for index in range(len(lineList)):
                        if index in keeperIndices:
                            keeperList.append(lineList[index])
                    outFile.write(",".join(keeperList) + "\n")
     
            i = i + 1

#FIXME: THIS ISN"T WORKING -- the burn isn't picking up any . . . 
getColumns("dailyFlowColumnsRefBurn.csv", True)
getColumns("dailyFlowColumnsRefNONBurn.csv", False)

