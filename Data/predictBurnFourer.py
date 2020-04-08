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
import scipy.stats.stats as stats
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

catchmentCharacteristicsFilePath = "corrected_catchment_characteristics.csv"
burnFilePath = "catchments_for_Q_analysis_corrected.csv"

flowOutputFile = "ref_test_flow_output.csv"
statsOutputFile = "ref_test_stats_output.csv"
burnPredictionFile = "ref_burn_prediction_output.csv"

with open(burnPredictionFile, "w+") as oFile:
    oFile.write("testNum, dataType,catchmentId,yearOfBurn,IndexOfStart(sr=1972),IndexOfStop(sr=1972)\n") 

with open(statsOutputFile, "w+") as oFile:
    oFile.write("testNum,sourceCatchment,targetCatchment,predictedSope,sourceSlope,predictedIntercept,sourceIntercept,predictedRValue,sourceRValue,predictedPValue,sourcePValue,predictedStdErr,sourceStdErr\n")

with open(flowOutputFile, "w+") as oFile:
    oFile.write("testNum,dataType,catchmentId,FlowValues\n")

numYearsToExclude = 10
startYear = 1972

numCatchmentsToCompare = 600
numChunksPerComparison = 2
numChunksPerValidation = 1
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
        nonComparableIndices = []

        minLength = min(len(self.sourceCatchmentFlow),len(self.targetCatchmentFlow))
        indicesToCheck = minLength - chunkLength - 20 
        if indicesToCheck < 0:
            indicesToCheck = 0

        for i in range(indicesToCheck): # avoid the edge, where there is likely to be error

            # if the previous was already good, no need to verify the entire range -- just look one ahead
            if (i - 1) in comparableIndices:
                try:
                    float(self.sourceCatchmentFlow[i + chunkLength]) / 2
                    float(self.targetCatchmentFlow[i + chunkLength]) / 2
                    comparableIndices.append(i)
                    numContinuousValidDays += 1

                except:
                    nonComparableIndices.append(i)

            else:
                numContinuousValidDays = 0
            
                try:
                    for j in range(chunkLength + 1): # otherwise, check the entire region
                        float(self.sourceCatchmentFlow[i + j]) / 2
                        float(self.targetCatchmentFlow[i + j]) / 2

                    comparableIndices.append(i)
                    numContinuousValidDays += 1

                except:
                    nonComparableIndices.append(i)


        # randomize the indices
        print('comparables')
        print(comparableIndices)

        print('non-comparables')
        print(nonComparableIndices)

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


    def getRandomChunk(self):

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

    def getChunk(self, startIndex):

        ''' returns the specific discharge for a specified chunk of consecutive days '''

        endIndex = startIndex + self.chunkLength

        targetFlowSelection = self.targetCatchmentFlow[startIndex:endIndex]
        sourceFlowSelection = self.sourceCatchmentFlow[startIndex:endIndex]
        try:
            targetFlowSelection = [flow / self.targetCatchmentSize for flow in targetFlowSelection]
            sourceFlowSelection = [flow / self.sourceCatchmentSize for flow in sourceFlowSelection]
        except:
            print(startIndex)
            print(targetFlowSelection)
            print(sourceFlowSelection)

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
                    
                    if catchment in self.catchments: # allows me to use reference/non-reference easily
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
                #try:
                    in1 = np.asarray(sourceCatchmentFlow[i:i + chunkLength]).astype(float)
                    in2 = np.asarray(targetCatchmentFlow[i:i + chunkLength]).astype(float)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(in1,in2)
                #except:
                #    print("***********************************************************************************")
                #    print("***********************************************************************************")
                #    print("***********************************************************************************")
                #    print("***********************************************************************************")
                #    print(sourceCatchmentFlow[i:i + chunkLength])
                #    print(targetCatchmentFlow[i:i + chunkLength])
                #    return False

                #if r_value*r_value > 0.60:
                    numComparableDays += 1

        if numComparableDays >= 3:
            return True
        
        return False
        

    def catchmentsComparable(self, sourceCatchment, targetCatchment):
        
        sourceTestFlow = getFlow(sourceCatchment, trainingDays, -1, self.burnChecker)
#        sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, self.burnChecker)
        sourceTrainFlow = getFlow(sourceCatchment, 0, trainingDays, self.burnChecker)
        
        targetTestFlow = getFlow(targetCatchment, trainingDays, -1, self.burnChecker)
#        targetValFlow = getFlow(targetCatchment, trainingDays, valDays, self.burnChecker)
        targetTrainFlow = getFlow(targetCatchment, 0, trainingDays, self.burnChecker)

        testComparable = self.flowsComparable(sourceTestFlow, targetTestFlow)
#        valComparable = self.flowsComparable(sourceValFlow, targetValFlow)
        trainComparable = self.flowsComparable(sourceTrainFlow, targetTrainFlow)
        

        if testComparable and trainComparable: # and valComparable:
            return True
        else:
            return False
    
    def getFlowCorrelation(self, flow1, flow2):
        minLength = min(len(flow1),len(flow2))
        indicesToCheck = minLength - chunkLength * 2
        
        maxNumValidDays = 0
        numContinuousValidDays = 0

        for i in range(indicesToCheck): 

            if (flow1[i] == None) or (flow2[i] == None):
                numContinousValidDays = 0
            else:
                numContinuousValidDays += 1
                if numContinuousValidDays > maxNumValidDays:
                    maxNumValidDays = numContinuousValidDays

            if numContinuousValidDays == 365:
        
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.asarray(flow1[i - 364:i]).astype(float), np.asarray(flow2[i - 364:i]).astype(float))
                return (r_value * r_value)

        return -1


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
                correlations.append(-1.0)
            elif self.catchmentsComparable(sourceCatchment, targetCatchment):
                srcFlow = getFlow(sourceCatchment, 0, trainingDays, self.burnChecker)
                targetFlow = getFlow(targetCatchment, 0, trainingDays, self.burnChecker)
                correlation = self.getFlowCorrelation(srcFlow, targetFlow)
                correlations.append(correlation)
            else:
                correlations.append(-1.0)
#            except:
                #print('c')
                #correlations.append(-1.0)
                #print(str(targetCatchment) + " burned catchment didn't exist in list of catchments")

        if len(correlations) == 0:
            return -1

        print(correlations) 
        maxIndex = correlations.index(max(correlations))
        print(maxIndex)

        if correlations[maxIndex] == -1:
            return -1

        sourceCatchment = self.catchments[targetIndex + comparisonIndices[maxIndex]]
        print()
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
        sourceFlowChunk, targetFlowChunk, startIndex = valChunker.getRandomChunk()

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



def predictBurn(model, sourceFlow, targetFlow, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, burnYear,start,stop, predictFile, trainChunker, numTestsRun):
    print('in here')
    
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

    # split it up into k-mers
    # predict each k-mer 
    # average across the predictions
    # save those averages
    k = 365
    numKmers = len(sourceFlow) - k - 1
    
    
    # because we're not using the flowChunker, we need to make specific discharge manually
    for i in range(len(sourceFlow)):
        try:
            sourceFlow[i] = sourceFlow[i] / float(sourceCatchmentSize)
        except:
            pass
    for i in range(len(targetFlow)):
        try:
            targetFlow[i] = targetFlow[i] / float(targetCatchmentSize)
        except:
            pass
    
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

        kmer = sourceFlow[kstart:kend]
        modelInput = np.concatenate((preChunk, kmer), axis=0) 
        predictedKmer = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0)) 
        predictedKmer = predictedKmer.squeeze().cpu().detach().numpy()

        predictedKmer = [max(f,0) for f in predictedKmer]
       
        preFluff = [""] * (kstart)
        postFluff = [""] * (len(sourceFlow) - len(preFluff) - k)
        
        #(numKmers - kend - 1)
        
        predictedKmer = preFluff + predictedKmer + postFluff
        # make a pre-kmer
        # make a post-kmer
        # add them all together
        # write to the file


        # record the information

        predictedFlow = [str(f) for f in predictedKmer]
        
        print("lengths")
        print(len(predictedFlow))
        print(len(sourceFlow))
        print(len(targetFlow))

        with open(predictFile, "a") as oFile:
            
            predictedLine = [str(numTestsRun),"predicted", str(targetCatchment), str(burnYear), str(start), str(stop)]
            predictedLine = predictedLine + predictedFlow
            predictedLine = ",".join(predictedLine)     

            oFile.write(predictedLine + '\n')


def testModel(model, testChunker, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, objective, flowOutputFile, statsOutputFile, trainChunker):
    gc.collect()
    model.eval()
    losses = []

    sourceCatchmentSize = sourceCharacteristics[1]
    targetCatchmentSize = targetCharacteristics[1]

    preChunk = trainChunker.getPreChunk()
        

    comparableIndices = testChunker.comparableIndices 
    comparableIndices.sort()
    predRSqrd = []
    sourceRSqrd = []
    

    for startIndex in comparableIndices:
        # get a chunk
        sourceFlowChunk, targetFlowChunk, startIndex = testChunker.getChunk(startIndex)

        modelInput = np.concatenate((preChunk, sourceFlowChunk), axis=0)
        

        predictedFlowChunk = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0))
        print("direct output")
        print(predictedFlowChunk)
        predictedFlowChunk = predictedFlowChunk.squeeze().cpu().detach().numpy()
    
        # remove anything less than 0
        sourceFlow = [max(f,0) for f in sourceFlowChunk]
        targetFlow = [max(f,0) for f in targetFlowChunk]
        predictedFlow = [max(f,0) for f in predictedFlowChunk]
        

        predSlope, predIntercept, predR_value, predP_value, predStd_err = stats.linregress(np.asarray(targetFlow).astype(float), np.asarray(predictedFlow).astype(float))
        sourceSlope, sourceIntercept, sourceR_value, sourceP_value, sourceStd_err = stats.linregress(np.asarray(targetFlow), np.asarray(sourceFlow))
        
        predRSqrd.append(predR_value*predR_value)
        sourceRSqrd.append(sourceR_value*sourceR_value)

        # write the regression info to file

        with open(statsOutputFile, "a") as soFile:
            newList = [str(numTestsRun), str(sourceCatchment), str(targetCatchment), predSlope, sourceSlope, predIntercept, sourceIntercept, predR_value, sourceR_value, predP_value, sourceP_value, predStd_err, sourceStd_err]
            newList = [str(value) for value in newList]
            newLine = ",".join(newList) + "\n"
            soFile.write(newLine)


        # write the flow info to file
        sourceFlow = [str(f) for f in sourceFlowChunk]
        targetFlow = [str(f) for f in targetFlowChunk]
        predictedFlow = [str(f) for f in predictedFlow]

        with open(flowOutputFile, "a") as oFile:
            sourceLine = [str(numTestsRun),"source", str(sourceCatchment)]
            sourceLine = sourceLine + sourceFlow
            sourceLine = ",".join(sourceLine)

            targetLine = [str(numTestsRun),"target", str(targetCatchment)]
            targetLine = targetLine + targetFlow
            targetLine = ",".join(targetLine)

            predictedLine = [str(numTestsRun),"predicted", str(targetCatchment)]
            predictedLine = predictedLine + predictedFlow
            predictedLine = ",".join(predictedLine)     

            oFile.write(sourceLine  + '\n')
            oFile.write(predictedLine + '\n')
            oFile.write(targetLine + '\n')

    model.train()
    gc.collect()
    if sum(predRSqrd) > sum(sourceRSqrd):
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

catchments = []
with open(catchmentCharacteristicsFilePath, "r+") as flowFile:
    i = 0
    for line in flowFile:
        if i > 0:
            lineList = line.split(",")
            catchment = lineList[0]
            if len(catchment) == 7:
                catchment = "0" + catchment
            if not catchment in referenceCatchments:
                catchments.append(catchment)
        i = i + 1

picker = CatchmentPicker(catchments, burnFilePath, numYearsToExclude, startYear)

# train!

trainingLosses = []
validationLosses = []

print(picker.burnedCatchments)

for targetCatchment in picker.burnedCatchments:
    #try:
        if len(targetCatchment) == 7:
            targetCatchment = "0" + targetCatchment

        sourceCatchment = picker.getCorrelatedCatchment(targetCatchment)

        numTestsRun += 1

        if not sourceCatchment == -1:
            if len(sourceCatchment) == 7:
                sourceCatchment = "0" + sourceCatchment


            tweakState = TweakState()
            tweakState.currentState["activation"] = "selu"
            tweakState.currentState["batchNorm"] = "off"
            tweakState.currentState["labelSmoothing"] = "off"
            tweakState.currentState["learningRate"] = "clr"
            tweakState.currentState["regularization"] = "dropout"
            tweakState.currentState["initialization"] = "orthogonal"

            # reset the model
            model = ResNet(tweakState.resParams, tweakState.currentState)
            last_saved_iteration = 3000
            model.load_state_dict(torch.load("big_res_model-" + str(last_saved_iteration)))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.cuda()

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

            preChunk = trainChunker.getPreChunk()

            for chunk in range(numChunksPerComparison):

                optimizer.zero_grad()

                # get a chunk
                sourceFlowChunk, targetFlowChunk, startIndex = trainChunker.getRandomChunk()
                # initiate the hiddens
                
                print('mean')
                print(np.mean(preChunk))
                print(np.mean(sourceFlowChunk))
                print(np.mean(targetFlowChunk))

                modelInput = np.concatenate((preChunk, sourceFlowChunk), axis=0)
                modelInput = makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
                predictedFlowChunk = model(modelInput)
                print('train')
                print("the numbers you want")
                print(predictedFlowChunk.squeeze().shape)
                print(makeTensor(targetFlowChunk).squeeze().shape)
                print(torch.mean(predictedFlowChunk.squeeze()))
                print(torch.mean(makeTensor(targetFlowChunk).squeeze()))
                loss = objective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze())
                print(loss)
                loss1 = smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze(), 24)
                print(loss1)
                loss2 = smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze(), 12)
                print(loss2)
                loss3 = smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze(), 7)
                print(loss3)
                loss4 = smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze(), 3)
                print(loss4)
                loss5 = fourierObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze())
                print("loss")
                print(loss5)

                loss.backward()
                optimizer.step()

                modelInput = np.concatenate((preChunk, sourceFlowChunk), axis=0)
                predictedFlowChunk = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0))
                print('after a step')
                print(modelInput.shape)
                print(modelInput)
                print(predictedFlowChunk)


                sourceFlow = [float(f) for f in sourceFlowChunk]
                targetFlow = [float(f) for f in targetFlowChunk]

                predictedFlowChunk = predictedFlowChunk.squeeze().cpu().detach().numpy()
   
                np.save("pred_non_ref_sourceFlow",sourceFlow)
                np.save("pred_non_ref_targetFlow",targetFlow)
                np.save("pred_non_ref_preditedFlow", predictedFlowChunk)


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
                    won = testModel(model, testChunker, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, objective, flowOutputFile, statsOutputFile,trainChunker)
    
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

                        predictBurn(model, sourceBurnFlow, targetBurnFlow, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, burnYear, preStart, stop, burnPredictionFile, trainChunker, numTestsRun)

            print("number of wins  : " + str(numWins))
            print("number of losses: " + str(numLosses))
        
    #except:
    #    print('catchment comparison failed')

print("final number of wins  : " + str(numWins))
print("final number of losses: " + str(numLosses))

#    if i % modelSaveInterval == 0:
#        torch.save(model.state_dict(), 'model-' + str(i))
#        pickle.dump(trainingLosses, open(("trainingLosses-" + str(i)), "wb"))
#        pickle.dump(validationLosses, open(("validationLosses-" + str(i)), "wb"))
# extract the flow data for each
# remove flow data that is burned or that is reserved for validation and testing ** save the last 19% of valid comparison days (remove anything after Jan 1 1912
#

