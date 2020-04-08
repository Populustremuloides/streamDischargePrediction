import matplotlib.pyplot as plt
import numpy as np


rootFolder = "/home/sethbw/Documents/brian_flow_code/Data"
inFilePath = rootFolder + "/compressed_ref_burn_prediction.csv"

saveFolder = "/home/sethbw/Documents/brian_flow_code/predictedFigures"

def makeGraph(prediced, source, target, targetCatchment, sourceCatchment):
    plt.plot(target, label="targetFlow-" + targetCatchment)
    plt.plot(source, label="sourceFlow-" + sourceCatchment)
    plt.plot(predicted, label="predictedFlow-" + targetCatchment)

    plt.legend()
    
    fileName = "/" + targetCatchment + sourceCatchment
    filePath = saveFolder + fileName
    plt.savefig(filePath)
    plt.show()
    plt.clf()

with open(inFilePath, "r+") as inFile:
    i = 0
    target = []
    source = []
    predicted = []
    for line in inFile:
        if i > 0:
            line = line.split(",")
            flow = line[6:-1]
            flow = [day.replace("\\n","") for day in flow]

            for index in range(len(flow)):
                try:
                    flow[index] = float(flow[index])
                except:
                    pass
                    #flow = [float(day) for day in flow]
            
            if i % 3 == 0:
                # save the target
                target = flow
                targetCatchment = line[3]
                
            elif i % 3 == 1:
                # save the source
                if i > 1:
                    predicted = flow
                    makeGraph(predicted, source, target, targetCatchment, sourceCatchment)

            elif i % 3 == 2:
                # save the prediction
                # make the graph 
        
                source = flow
                sourceCatchment = line[3]
        i = i + 1
