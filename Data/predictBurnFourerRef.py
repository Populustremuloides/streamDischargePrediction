import sys
import random
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from YearModels import *


# go through every burned catchment
# find the most correlated catchment we can
# remove the burned data
# train on that catchment
# predict the burned time: 2 years before to 2 years after
# record that in a file

startYear = 1972

def indexToYear(index, startYear):
    '''start year must be a leap year!'''
    count = 0
    numTimesAdded = 0
    year = startYear
    indexNotReached = True
    
    while indexNotReached:
        if numTimesAdded % 4 == 0:
            count += 366
            year += 1
        else:
            count += 365
            year += 1
        if count >= index:
            indexNotReached = False

    return year
    
class FourrierLoss(nn.Module):
    def __init__(self):
        super(FourrierLoss, self).__init__()
        self.objective = nn.MSELoss()

    def getS(self, inputTens, k):
        ''' computes a 1D smoothness with window size k across an iput tensor '''
        smoothnesses = []
        for i in range(inputTens.shape[0] - k - 1):
            startIndex = i
            endIndex = startIndex + k

            kmer = inputTens[startIndex:endIndex]
            smoothness = self.computeSmoothness(kmer)
            smoothnesses.append(smoothness)
        return torch.FloatTensor(smoothnesses)

    def forward(self, yHat, y):
        yHatF = torch.rfft(yHat, 1, onesided = False)
        yF = torch.rfft(y, 1, onesided = False)

        yHatF = yHatF.reshape(-1)
        yF = yF.reshape(-1)

        return self.objective(yHatF, yF)

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()
        self.objective = nn.MSELoss()

    def computeAvgSmoothness(self, kmer):
        ''' average smoothness is the average difference from the mean '''
        mean = torch.mean(kmer)
        kMerMinusMean = kmer - mean
        absDiffs = torch.abs(kMerMinusMean)
        sumOfDiffs = torch.sum(absDiffs)
        return sumOfDiffs / kmer.shape[0]

    def computeSmoothness(self, kmer):
        ''' smoothness is the sum of differences from the mean of the tensor'''
        mean = torch.mean(kmer)
        kMerMinusMean = kmer - mean
        absDiffs = torch.abs(kMerMinusMean)
        sumOfDiffs = torch.sum(absDiffs)
        return sumOfDiffs

    def getSmoothness1D(self, inputTens, k):
        ''' computes a 1D smoothness with window size k across an iput tensor '''
        smoothnesses = []
        for i in range(inputTens.shape[0] - k - 1):
            startIndex = i
            endIndex = startIndex + k

            kmer = inputTens[startIndex:endIndex]
            smoothness = self.computeSmoothness(kmer)
            smoothnesses.append(smoothness)
        return torch.FloatTensor(smoothnesses)

    def forward(self, yHat, y, k):
        yHatSmooth = self.getSmoothness1D(yHat, k)
        ySmooth = self.getSmoothness1D(y, k)

        return self.objective(yHatSmooth, ySmooth)

smoothObjective = SmoothnessLoss()
fourierObjective = FourrierLoss()


catchmentCharacteristicsFilePath = "normalized_zerod_catchment_characteristics_corrected.csv"
burnFilePath = "catchments_for_Q_analysis_corrected.csv"

outputFile = "ref_val_output.csv"
burnPredictionFile = "ref_burn_prediction_output.csv"

with open(burnPredictionFile, "w+") as oFile:
    oFile.write("testNum, dataType,catchmentId,yearOfBurn,IndexOfStart(sr=1972),IndexOfStop(sr=1972)\n") 

with open(outputFile, "w+") as oFile:
    oFile.write("testNum,dataType,correlationToTarget,catchmentId,FlowValues\n")

numYearsToExclude = 10
startYear = 1972

numCatchmentsToCompare = 1000
numChunksPerComparison = 1000
numChunksPerValidation = 10
numChunksPerTest = 1
trainBurnIn = 0
burnIn = 0 + trainBurnIn
chunkLength = 365 + burnIn

trainingDays = 10000 # 10227
#valDays = trainingDays + (3000)
testDays = trainingDays + (7000)


tweakState = TweakState()
tweakState.currentState["activation"] = "selu"
tweakState.currentState["batchNorm"] = "on"
tweakState.currentState["labelSmoothing"] = "off"
tweakState.currentState["learningRate"] = "clr"
tweakState.currentState["regularization"] = "dropout"
tweakState.currentState["initialization"] = "orthogonal"


model = ResNet(tweakState.resParams, tweakState.currentState) 

last_saved_iteration = 3000
model.load_state_dict(torch.load("big_res_model-" + str(last_saved_iteration)))
model = model.cuda() 

numLosses = 0
numWins = 0

for parameter in model.parameters():
    parameter.requires_grad = True

#modelSaveInterval = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(torch.load('big_res_optimizer-' + str(last_saved_iteration)))
optimizer.state_dict()['lr'] = 0.001

numTestsRun = 0

objective = nn.MSELoss()


# ***************************************************** Functions and Classes


def getCatchmentCharacteristics(catchmentToGet):
    with open(catchmentCharacteristicsFilePath, "r+") as catchmentFile:
        for line in catchmentFile:
            characteristics = line.split(",")
            catchment = characteristics[0]
            if len(catchment) == 7:
                catchment = "0" + catchment
            if catchment == catchmentToGet:
                return characteristics
    return "NOT FOUND"


def getBurnFlow(catchmentToGet, startIndex, stopIndex):
#    try:
        flowDir = "/home/sethbw/Documents/brian_flow_code/Data/all_flow"
        flowPath = os.path.join(flowDir, (catchmentToGet + ".npy"))
        flow = np.load(flowPath, allow_pickle=True)
        flow.tolist()
        return flow[startIndex:stopIndex]


def getFlow(catchmentToGet, startIndex, stopIndex, burnChecker):
#    try:
        flowDir = "/home/sethbw/Documents/brian_flow_code/Data/all_flow"
        flowPath = os.path.join(flowDir, (catchmentToGet + ".npy"))
        flow = np.load(flowPath, allow_pickle=True)
        flow.tolist()
        flow = burnChecker.removeBurnedData(flow, catchmentToGet)
        print('len flow') # 17515
        print(len(flow))
        return flow[startIndex:stopIndex]

#    except:
#        print("exception occured while getting flow")
#        return []


class BurnChecker():
    def __init__(self, burnFilePath, numYearsToExclude, startYear):
        self.burnFilePath = burnFilePath
        self.numYearsToExclude = numYearsToExclude
        self.startYear = startYear

        self.catchmentToBurnedDays = {}
        self.initializeCatchmentToBurnedDays()

    def initializeCatchmentToBurnedDays(self):
        with open(self.burnFilePath) as burnFile:
            i = 0
            for line in burnFile:
                if i > 0:
                    data = line.split(",")
                    catchment = data[0]
                    
                    if len(catchment) == 7:
                        catchment = "0" + catchment

                    startY = data[1]
                    startDate = startY + "-01-01"
                    endY = int(startY) + self.numYearsToExclude + 1
                    endDate = str(endY) + "-01-01"
                    startIndex = self.dateToIndex(startDate, self.startYear)
                    endIndex = self.dateToIndex(endDate, self.startYear)
                    if catchment in self.catchmentToBurnedDays.keys():
                        self.catchmentToBurnedDays[catchment].append((startIndex, endIndex))
                    else:
                        self.catchmentToBurnedDays[catchment] = []
                        self.catchmentToBurnedDays[catchment].append((startIndex, endIndex))
                i = i + 1

    def dateToIndex(self, date, startYear):
        year, month, day = date.split("-")
        year = int(year)
        month = int(month)
        day = int(day)

        index = 0

        # add the years *************************************************
        numYears = year - startYear
        for yearSinceStart in range(numYears):
            currentYear = yearSinceStart + startYear
            if currentYear % 4 == 0:
                daysInYear = 366
            else:
                daysInYear = 365

            index = index + daysInYear

        # add the months ***********************************************

        if year % 4 == 0:  # if it is a leap year
            monthToDays = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        else:
            monthToDays = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

        monthVal = int(month)

        if monthVal > 1:
            for month in range(monthVal - 1):  # don't include the current month because it isn't over yet!
                month = month + 1  # start in January, not month 0
                index = index + monthToDays[month]

        # add the days *******************************************
        index = index + day

        return index

    def removeBurnedData(self, flow, catchmentToGet):
        if catchmentToGet in self.catchmentToBurnedDays.keys():
            burnedDays = self.catchmentToBurnedDays[catchmentToGet]

            for index in range(len(flow)):
                for startIndex, endIndex in burnedDays:
                    if index >= startIndex and index < endIndex:
                        flow[index] = None
        return flow




class FlowChunker():

    def __init__(self, sourceCatchmentFlow, targetCatchmentFlow, chunkLength, sourceCatchmentSize, targetCatchmentSize):

        self.sourceCatchmentSize = float(sourceCatchmentSize)
        self.targetCatchmentSize = float(targetCatchmentSize)

        self.chunkLength = chunkLength
        self.startIndicesToLoss = {}
        self.alpha = 100000

        self.targetCatchmentFlow = targetCatchmentFlow
        self.sourceCatchmentFlow = sourceCatchmentFlow

        self.comparableIndices = self.getComparableIndices()

        # add the first chunk
        newKey = self.comparableIndices[0]
        self.startIndicesToLoss[newKey] = int(1000)
    def getComparableIndices(self):
        comparableIndices = []

        minLength = min(len(self.sourceCatchmentFlow),len(self.targetCatchmentFlow))
        indicesToCheck = minLength - chunkLength - 20 
        if indicesToCheck < 0:
            indicesToCheck = 0

        for i in range(indicesToCheck): # avoid the edge, where there is likely to be error
           
            # if the previous was already good, no need to verify the entire range -- just look one ahead
            if (i - 1) in comparableIndices:
                if not (self.sourceCatchmentFlow[i + chunkLength] == None) and not (self.targetCatchmentFlow[i + chunkLength] == None):
                    comparableIndices.append(i)

            else:
                numContinuousValidDays = 0
                # examine every part of the chunk length to verify it is non-zero
                for j in range(chunkLength):
                    if not (self.sourceCatchmentFlow[i + j] == None) and not (self.targetCatchmentFlow[i + j] == None):
                        numContinuousValidDays += 1
     
                if numContinuousValidDays == chunkLength:
                    comparableIndices.append(i)
        
        # randomize the indices
        random.shuffle(comparableIndices)

        return comparableIndices

    def clear_to_add(self):

        for key in self.startIndicesToLoss.keys():
            if self.startIndicesToLoss[key] > self.alpha:
                return False

        return True

    def add_new_random_num(self):

        if self.clear_to_add():
            newIndexNotAdded = False

            if len(self.startIndicesToLoss) < len(self.comparableIndices):
                newIndex = self.comparableIndices[len(self.startIndicesToLoss)]
                self.startIndicesToLoss[newIndex] = int(1000)

    def update_loss(self, startIndex, loss):
        self.startIndicesToLoss[startIndex] = loss

    def getPreChunk(self):
        ''' returns the specific discharge for the first comparable chunk of consecutive days, concatenated together for both source and target '''

        keys = list(self.startIndicesToLoss.keys())
        startIndex = keys[0]
        endIndex = startIndex + self.chunkLength

        targetFlowSelection = self.targetCatchmentFlow[startIndex:endIndex]
        sourceFlowSelection = self.sourceCatchmentFlow[startIndex:endIndex]

        targetFlowSelection = [flow / self.targetCatchmentSize for flow in targetFlowSelection]
        sourceFlowSelection = [flow / self.sourceCatchmentSize for flow in sourceFlowSelection]

        return np.concatenate((targetFlowSelection, sourceFlowSelection), axis=0)
    def getChunk(self):

        ''' returns the specific discharge for a random comparable chunk of consecutive days '''

        self.add_new_random_num()
        keys = list(self.startIndicesToLoss.keys())
        randomStartIndex = random.randint(1, len(self.startIndicesToLoss.keys()) - 1)

        startIndex = keys[randomStartIndex]
        endIndex = startIndex + self.chunkLength

        targetFlowSelection = self.targetCatchmentFlow[startIndex:endIndex]
        sourceFlowSelection = self.sourceCatchmentFlow[startIndex:endIndex]

        targetFlowSelection = [flow / self.targetCatchmentSize for flow in targetFlowSelection]
        sourceFlowSelection = [flow / self.sourceCatchmentSize for flow in sourceFlowSelection]

        return targetFlowSelection, sourceFlowSelection, startIndex


class CatchmentPicker():
    def __init__(self, catchments, burnFilePath, numYearsToExclude, startYear):
        self.catchments = catchments

        self.burnChecker = BurnChecker(burnFilePath, numYearsToExclude, startYear)

        self.burnedCatchments = []
        with open(burnFilePath) as burnFile:
            i = 0
            for line in burnFile:
                if i > 0:
                    data = line.split(",")
                    catchment = data[0]
                    print(catchment)
                    
                    if len(catchment) == 7:
                        catchment = "0" + catchment

                    self.burnedCatchments.append(catchment)

                i = i + 1

    def flowsComparable(self, sourceCatchmentFlow, targetCatchmentFlow):
        # check to see that they are correlated enough:

        numComparableDays = 0

        minLength = min(len(sourceCatchmentFlow),len(targetCatchmentFlow))
        indicesToCheck = minLength - chunkLength * 2 # make sure there are plenty of comparable indices 

        if indicesToCheck < 0:
            indicesToCheck = 0

        for i in range(indicesToCheck): # avoid the edge, where there is likely to be error

            numContinuousValidDays = 0
            # examine every part of the chunk length to verify it is non-zero
            for j in range(chunkLength):
                if not (sourceCatchmentFlow[i + j] == None) and not (targetCatchmentFlow[i + j] == None):
                    numContinuousValidDays += 1

            if numContinuousValidDays == (chunkLength):
                try:
                    correlation = pearsonr(sourceCatchmentFlow[i:i + chunkLength], targetCatchmentFlow[i:i + chunkLength])
                except:
                    print("***********************************************************************************")
                    print("***********************************************************************************")
                    print("***********************************************************************************")
                    print("***********************************************************************************")
                    print(sourceCatchmentFlow[i:i + chunkLength])
                    print(targetCatchmentFlow[i:i + chunkLength])
                    return False

                if correlation[0] > 0.60:
                    numComparableDays += 1
#                    print('winning correlation was: ' + str(correlation[0]))
#                    return True
#                else:
#                    print('correlation was: ' + str(correlation[0]))
#                    return False

        if numComparableDays >= 3:
            return True

        return False


    def catchmentsComparable(self, sourceCatchment, targetCatchment):
        
        
        sourceTestFlow = getFlow(sourceCatchment, 0, trainingDays, self.burnChecker)
#        sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, self.burnChecker)
        sourceTrainFlow = getFlow(sourceCatchment, valDays, -1, self.burnChecker)
    
        
        targetTestFlow = getFlow(targetCatchment, 0, trainingDays, self.burnChecker)
#        targetValFlow = getFlow(targetCatchment, trainingDays, valDays, self.burnChecker)
        targetTrainFlow = getFlow(targetCatchment, valDays, -1, self.burnChecker)

        testComparable = self.flowsComparable(sourceTestFlow, targetTestFlow)
#        valComparable = self.flowsComparable(sourceValFlow, targetValFlow)
        trainComparable = self.flowsComparable(sourceTrainFlow, targetTrainFlow)
        
        print("test")
        print(testComparable)
#        print('val')
#        print(valComparable)
        print('train')
        print(trainComparable)

        if testComparable and trainComparable: # and valComparable:
            return True
        else:
            return False

    def getCorrelatedCatchment(self, targetCatchment):
        # make sure the catchment is labeled correctly
        if len(targetCatchment) == 7:
            targetCatchment = "0" + targetCatchment
        
        # find the index of that catchment

        targetIndex = ""

        cntr = 0
        for catchment in self.catchments:
            if len(catchment) == 7:
                catchment = "0" + catchment
            if catchment == targetCatchment:
                targetIndex = cntr
            cntr += 1
        
        # look in the neighborhood for the most correlated catchment
        comparisonIndices = list(range(-15,15))
        correlations = [] 

        if targetIndex == "":
            print('targetCatchment not found: ')
            print(targetCatchment)

        for i in comparisonIndices:
#            try:
            sourceIndex = targetIndex + i
               
            sourceCatchment = self.catchments[sourceIndex]
            targetCatchment = self.catchments[targetIndex]
            
            if i == 0:
                print('a')
                correlations.append(-1.0)
            elif self.catchmentsComparable(sourceCatchment, targetCatchment):
                srcFlow = getFlow(sourceCatchment, 0, trainingDays, self.burnChecker)
                targetFlow = getFlow(targetCatchment, 0, trainingDays, self.burnChecker)
                correlation = self.getFlowCorrelation(srcFlow, targetFlow)
                correlations.append(correlation)
            else:
                print('b')
                correlations.append(-1.0)
#            except:
                #print('c')
                #correlations.append(-1.0)
                #print(str(targetCatchment) + " burned catchment didn't exist in list of catchments")

        if len(correlations) == 0:
            print('d')
            return -1

        print(correlations) 
        maxIndex = correlations.index(max(correlations))
        print(maxIndex)

        if correlations[maxIndex] == -1:
            return -1

        sourceCatchment = self.catchments[targetIndex + comparisonIndices[maxIndex]]

        print(sourceCatchment)

        return sourceCatchment



def evaluateModel(model, valChunker, sourceCharacteristics, targetCharacteristics, objective, trainChunker):
    gc.collect()
    model.eval()
    losses = []

    sourceCatchmentSize = sourceCharacteristics[1]
    targetCatchmentSize = targetCharacteristics[1]

    preChunk = trainChunker.getPreChunk()

    for _ in range(numChunksPerValidation):

        totalLoss = 0

        # get a chunk
        sourceFlowChunk, targetFlowChunk, startIndex = valChunker.getChunk()

        predictedFlow = []

        modelInput = np.concatenate((preChunk, sourceFlowChunk), axis=0)
        predictedFlowChunk = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0))
        loss = objective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())
#        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 48)
#        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 28)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 24)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 12)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 7)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 3)
        loss += fourierObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())

        losses.append(loss.item())


    sourceFlow = [float(f) for f in sourceFlowChunk]
    targetFlow = [float(f) for f in targetFlowChunk]
    predictedFlow = [float(f) for f in predictedFlow]

    predictedFlowChunk = predictedFlowChunk.squeeze().cpu().detach().numpy()

    np.save("sourceFlow",sourceFlow)
    np.save("targetFlow",targetFlow)
    np.save("preditedFlow", predictedFlow)

#    plt.plot(sourceFlow, label="sourceFlow")
#    plt.plot(targetFlow, label="targetFlow")
#    plt.plot(predictedFlow, label="predictedFlow")
#    plt.legend()
#    plt.show()

    model.train()
    gc.collect()
    return torch.mean(torch.FloatTensor(losses))



def predictBurn(model, sourceFlow, targetFlow, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, burnYear,start,stop,predictFile, trainChunker):
    print('in here')
    
    numTestsRun += 1

    gc.collect()
    model.eval()

    predictedFlow = []
    # initiate the hiddens


    sourceCatchmentSize = sourceCharacteristics[1]
    targetCatchmentSize = targetCharacteristics[1]

    preChunk = trainChunker.getPreChunk()

    # for char in chunk:
    output = targetFlow[0]
    predictedFlow.append(output)

    # FIXME: Predict burn 

    # split it up into k-mers
    # predict each k-mer 
    # average across the predictions
    # save those averages
    k = 365
    numKmers = len(sourceFlow) - k - 1

    
    # write the header
    with open(predictFile, "a") as oFile:
        for flw in targetFlow:
            if flw == None:
                flw = ""

        sourceFlowStr = [str(f) for f in sourceFlow]
        targetFlowStr = [str(f) for f in targetFlow]

        sourceLine = [str(numTestsRun),"source", str(sourceCatchment), str(burnYear), str(start), str(stop)]
        sourceLine = sourceLine + sourceFlowStr
        sourceLine = ",".join(sourceLine)

        targetLine = [str(numTestsRun),"target", str(targetCatchment), str(burnYear), str(start), str(stop)]
        targetLine = targetLine + targetFlowStr
        targetLine = ",".join(targetLine)

        oFile.write(sourceLine  + '\n')
        oFile.write(targetLine + '\n')

    # write the predictions
    for numKmer in range(numKmers):
        kstart = numKmer
        kend = kstart + k

        kmer = sourceFlow[ksrtart:kend]
        modelInput = np.concatenate((preChunk, kmer), axis=0)
        predictedKmer = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0)) 
        predictedKmer = [max(f,0) for f in predictedKmer]
       
        preFluff = [""] * (kstart)
        postFluff = [""] * (numKmers - kend - 1)
        
        predictedKmer = preFluff + predictedKmer + postFluff
        # make a pre-kmer
        # make a post-kmer
        # add them all together
        # write to the file


        # record the information

        predictedFlow = [str(f) for f in predictedKmer]
    
        with open(predictFile, "a") as oFile:
            predictedLine = [str(numTestsRun),"predicted", str(targetCatchment), str(burnYear), str(start), str(stop)]
            predictedLine = predictedLine + predictedFlow
            predictedLine = ",".join(predictedLine)     

            oFile.write(predictedLine + '\n')


def testModel(model, testChunker, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, objective, outputFile):
    gc.collect()
    model.eval()
    losses = []

    sourceCatchmentSize = sourceCharacteristics[1]
    targetCatchmentSize = targetCharacteristics[1]

    preChunk = trainChunker.getPreChunk()

    for _ in range(numChunksPerTest):


        # get a chunk
        sourceFlowChunk, targetFlowChunk, startIndex = testChunker.getChunk()

        predictedFlow = []



        modelInput = np.concatenate((preChunk, sourceFlowChunk), axis=0)
        predictedFlowChunk = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0))
        loss = objective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())
#        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 48)
#        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 28)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 24)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 12)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 7)
        loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 3)
        loss += fourierObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())

        losses.append(loss.item())


        # remove anything less than 0

        sourceFlow = [max(f,0) for f in sourceFlowChunk]
        targetFlow = [max(f,0) for f in targetFlowChunk]
        predictedFlow = [max(f,0) for f in predictedFlow]
        

        # calculate the correlations
        predCorrelation = pearsonr(targetFlow, predictedFlow)
        predCorrelation = predCorrelation[0]
        realCorrelation = pearsonr(targetFlow, sourceFlow)
        realCorrelation = realCorrelation[0]

        # calculate the average error
        sourceDiff = torch.mean(targetFlow - sourceFlow).item()
        predDiff = torch.mean(targetFlow - predictedFlow).item()

        # calculate the average percent error

        sourcePercDiff = (sourceDiff / torch.mean(targetFlow)).item()
        predPercDiff = (predDiff / torch.mean(targetFlow)).item()

        # record the information

        sourceFlow = [str(f) for f in sourceFlowChunk]
        targetFlow = [str(f) for f in targetFlowChunk]
        predictedFlow = [str(f) for f in predictedFlow]


        with open(outputFile, "a") as oFile:
            sourceLine = [str(numTestsRun),"source", str(realCorrelation), str(sourceCatchment), sourceDiff, sourcePrecDiff]
            sourceLine = sourceLine + sourceFlow
            sourceLine = ",".join(sourceLine)

            targetLine = [str(numTestsRun),"target", "1", str(targetCatchment), "0"]
            targetLine = targetLine + targetFlow
            targetLine = ",".join(targetLine)

            predictedLine = [str(numTestsRun),"predicted", str(predCorrelation), str(targetCatchment), targetDiff, predPercDiff]
            predictedLine = predictedLine + predictedFlow
            predictedLine = ",".join(predictedLine)     

            oFile.write(sourceLine  + '\n')
            oFile.write(predictedLine + '\n')
            oFile.write(targetLine + '\n')

    model.train()
    gc.collect()
    if sourceDiff > predDiff:
        return True
    else:
        return False


def makeTensor(flowString):
    try:
        flowFloat = [float(a) for a in flowString]
        
    except:
        print('exception in makeFlowTensor:')
        flowFloat = [0.0] * len(flowString)

    flowTensor = torch.FloatTensor(flowFloat)

    return flowTensor.cuda()


def makeInputTensor(flowString, targetCatchmentChars, sourceCatchmentChars):
    try:
        flowFloat = [float(a) for a in flowString]
        targFloat = [float(a) for a in targetCatchmentChars]
        sourceFloat = [float(a) for a in sourceCatchmentChars]

        flowFloat = flowFloat + targFloat + sourceFloat
    except:
        print('exception in makeFlowTensor:')
        flowFloat = 0.0

    flowTensor = torch.FloatTensor(flowFloat).unsqueeze(dim=0).unsqueeze(dim=0)

    return flowTensor.cuda()


# ************************************************ Action code starts here

# get a list of catchments present in flowFile:
catchments = []

with open(catchmentCharacteristicsFilePath, "r+") as flowFile:
    i = 0
    for line in flowFile:
        if i > 0:
            lineList = line.split(",")
            catchment = lineList[0]
            if len(catchment) == 7:
                catchment = "0" + catchment
            catchments.append(catchment)
        i = i + 1


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


picker = CatchmentPicker(referenceCatchments, burnFilePath, numYearsToExclude, startYear)

# train!

trainingLosses = []
validationLosses = []

print(picker.burnedCatchments)

for targetCatchment in picker.burnedCatchments:

    if len(targetCatchment) == 7:
        targetCatchment = "0" + targetCatchment

    sourceCatchment = picker.getCorrelatedCatchment(targetCatchment)

    if not sourceCatchment == -1:
        if len(sourceCatchment) == 7:
            sourceCatchment = "0" + sourceCatchment
   
        # reset the model
        model = GruFlowtron(numLayers=numLayers)
        stateDict = torch.load("model-19")
        model.load_state_dict(stateDict)
        model.cuda()

        last_saved_iteration = 3000
        model.load_state_dict(torch.load("big_res_model-" + str(last_saved_iteration)))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()

        # grab the training, test, and validaiton data

        sourceCharacteristics = getCatchmentCharacteristics(sourceCatchment)
        targetCharacteristics = getCatchmentCharacteristics(targetCatchment)

        sourceCatchmentSize = float(sourceCharacteristics[1])
        targetCatchmentSize = float(targetCharacteristics[1])

        foo = float(sourceCatchmentSize)
        foo = float(targetCatchmentSize)

        sourceTrainFlow = getFlow(sourceCatchment, 0, trainingDays, picker.burnChecker)
        targetTrainFlow = getFlow(targetCatchment, 0, trainingDays, picker.burnChecker)

#        sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, picker.burnChecker)
#        targetValFlow = getFlow(targetCatchment, trainingDays, valDays, picker.burnChecker)

        sourceTestFlow = getFlow(sourceCatchment, trainingDays, testDays, picker.burnChecker)
        targetTestFlow = getFlow(targetCatchment, trainingDays, testDays, picker.burnChecker)

        trainChunker = FlowChunker(sourceTrainFlow, targetTrainFlow, chunkLength, sourceCatchmentSize, targetCatchmentSize)
#        valChunker = FlowChunker(sourceValFlow, targetValFlow, chunkLength, sourceCatchmentSize, targetCatchmentSize)
        testChunker = FlowChunker(sourceTestFlow, targetTestFlow, chunkLength, sourceCatchmentSize, targetCatchmentSize)

        print()
        print()
        print("source catchment: " + str(sourceCatchment))
        print("target catchment: " + str(targetCatchment))
        print()

        loop = tqdm(total=numChunksPerComparison, position=0, leave=False)

        for chunk in range(numChunksPerComparison):

            optimizer.zero_grad()
            totalLoss = 0

            # get a chunk
            sourceFlowChunk, targetFlowChunk, startIndex = trainChunker.getChunk()
            # initiate the hiddens

            modelInput = np.concatenate((preChunk, sourceFlowChunk), axis=0)
            predictedFlowChunk = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0))
            loss = objective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())
            loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 24)
            loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 12)
            loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 7)
            loss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 3)
            loss += fourierObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())

            losses.append(loss.item())

            sourceFlow = [float(f) for f in sourceFlowChunk]
            targetFlow = [float(f) for f in targetFlowChunk]

            predictedFlowChunk = predictedFlowChunk.squeeze().cpu().detach().numpy()
   
            np.save("val_ref_sourceFlow",sourceFlow)
            np.save("val_ref_targetFlow",targetFlow)
            np.save("val_ref_preditedFlow", predictedFlowChunk)



#            if chunk % 5 == 0:
#                valLoss = evaluateModel(model, valChunker, sourceCharacteristics, targetCharacteristics, objective, trainChunker)
#                validationLosses.append((len(trainingLosses), valLoss))

#            trainChunker.update_loss(startIndex, totalLoss)
#            numChunks = str(len(trainChunker.startIndicesToLoss.keys()))
#            loop.set_description(
#                'comparison_num: {}, trainLoss: {:,.4f}, valLoss: {:,.4f}, num_chunks: {}'.format(i, totalLoss.item(), valLoss,numChunks))

            if chunk == (numChunksPerComparison - 1):
                # if the loss is > 50, redo it! (that way we can salvage the times where it failed)

                # test the accuracy on the 
                won = testModel(model, valChunker, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, objective, outputFile, trainChunker)

                if won:
                    numWins += 1
                    print("WON!")
                else:
                    numLosses += 1
                    print("lost")

                # predict burn! 

                burnChecker = BurnChecker(burnFilePath, numYearsToExclude, startYear)
                burns = burnChecker.catchmentToBurnedDays[targetCatchment]
                for burn in burns:
                    start = burn[0]
                    stop = burn[1]

                    burnYear = indexToYear(start, 1972)
                    preStart = None
                    if burnYear % 4 == 0:
                        preStart = start - 366
                        preBurnInStart = start - 366 - burnIn
                        targetBurnFlow = getBurnFlow(targetCatchment, preBurnInStart, stop)
                        sourceBurnFlow = getBurnFlow(sourceCatchment, preBurnInStart, stop)
                    else:
                        preStart = start - 365
                        preBurnInStart = start - 365 - burnIn
                        targetBurnFlow = getBurnFlow(targetCatchment, preBurnInStart, stop)
                        sourceBurnFlow = getBurnFlow(sourceCatchment, preBurnInStart, stop)
                    
                    # find out how far we can go with this catchment...
                    for index in range(len(sourceBurnFlow)):
                        if sourceBurnFlow[index] == None:
                            sourceBurnFlow = sourceBurnFlow[:index]
                            targetBurnFlow = targetBurnFlow[:index]
                            stop = index + preBurnInStart #FIXME
                            break

                    predictBurn(model, sourceBurnFlow, targetBurnFlow, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, burnYear, preStart, stop, burnPredictionFile, trainChunker)


            loop.update(1)
            totalLoss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            trainingLosses.append(totalLoss.item())

            print("number of wins  : " + str(numWins))
            print("number of losses: " + str(numLosses))


print("final number of wins  : " + str(numWins))
print("final number of losses: " + str(numLosses))

#    if i % modelSaveInterval == 0:
#        torch.save(model.state_dict(), 'model-' + str(i))
#        pickle.dump(trainingLosses, open(("trainingLosses-" + str(i)), "wb"))
#        pickle.dump(validationLosses, open(("validationLosses-" + str(i)), "wb"))
# extract the flow data for each
# remove flow data that is burned or that is reserved for validation and testing ** save the last 19% of valid comparison days (remove anything after Jan 1 1912
#

