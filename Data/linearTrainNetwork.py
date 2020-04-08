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



# maybe in the future use multiple catchments

catchmentCharacteristicsFilePath = "corrected_catchment_characteristics.csv"
burnFilePath = "catchments_for_Q_analysis.csv"

numYearsToExclude = 10
startYear = 1972

numCatchmentsToCompare = 100000
numChunksPerComparison = 1000
numChunksPerValidation = 10
trainBurnIn = 0
burnIn = 0 + trainBurnIn
chunkLength = 365 + burnIn

trainingDays = 10000 # 10227
valDays = trainingDays + 3000 # (trainingDays // 10)
#testDays = valDays +  #(trainingDays // 10)
flowCoefficient = 10000  # divide flow by this value to make the value less than 1 in most cases (and multiply the network output by this to get the real flow value!)


tweakState = TweakState()
tweakState.currentState["activation"] = "selu"
tweakState.currentState["batchNorm"] = "on"
tweakState.currentState["labelSmoothing"] = "off"
tweakState.currentState["learningRate"] = "clr"
tweakState.currentState["regularization"] = "dropout"
tweakState.currentState["initialization"] = "orthogonal"


model = ResNet(tweakState.resParams, tweakState.currentState)
model = model.cuda()


last_saved_iteration = 3100
model.load_state_dict(torch.load("big_res_model-" + str(last_saved_iteration)))
print(model)
model.train()
modelSaveInterval = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

objective = nn.MSELoss() # PercentageLoss() # nn.L1Loss() # PercentageLoss() #  nn.L1Loss()

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
        
        # for yHat and y:
            # for every k-mer
                # compute the smoothness value
                # store in a vector
        # take the MSE or L1 loss between the two

        yHatSmooth = self.getSmoothness1D(yHat, k)
        ySmooth = self.getSmoothness1D(y, k)
        
        return self.objective(yHatSmooth, ySmooth)

smoothObjective = SmoothnessLoss()
fourierObjective = FourrierLoss()

class PercentageLoss(nn.Module):
    def __init__(self,):
        super(PercentageLoss, self).__init__()

    def forward(self, yHat, y):
        diff = torch.abs(yHat - y)
        percentDiff = diff / y
        return (torch.sum(percentDiff) / y.shape[0]) / chunkLength



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

def getFlow(catchmentToGet, startIndex, stopIndex, burnChecker):
#    try:
        flowDir = "/home/sethbw/Documents/brian_flow_code/Data/all_flow"
        flowPath = os.path.join(flowDir, (catchmentToGet + ".npy"))

        #print(catchmentToGet)

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
                        flow[index] = ""
        return flow

# FIXME: I need to get the flowChunker to make the specific discharge. Then I need to undo that in maketensor. Then I need to plug in the right preFlowChunk and make sure it works with the new model.

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

    def flowsComparable(self, sourceCatchmentFlow, targetCatchmentFlow):
        # check to see that they are correlated enough:
        
        numComparableDays = 0

        minLength = min(len(sourceCatchmentFlow),len(targetCatchmentFlow))
        indicesToCheck = minLength - chunkLength * 2 # make sure there are plenty of comparable indices 

        if indicesToCheck < 0:
            indicesToCheck = 0
        print(indicesToCheck)
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
        sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, self.burnChecker)
        sourceTrainFlow = getFlow(sourceCatchment, valDays, -1, self.burnChecker)

        targetTestFlow = getFlow(targetCatchment, 0, trainingDays, self.burnChecker)
        targetValFlow = getFlow(targetCatchment, trainingDays, valDays, self.burnChecker)
        targetTrainFlow = getFlow(targetCatchment, valDays, -1, self.burnChecker)

        testComparable = self.flowsComparable(sourceTestFlow, targetTestFlow)
        valComparable = self.flowsComparable(sourceValFlow, targetValFlow)
        trainComparable = self.flowsComparable(sourceTrainFlow, targetTrainFlow)
        
        if testComparable and valComparable and trainComparable:
            return True
        else:
            return False


    def pickTwoCatchments(self, comparisonNum):

        catchmentsNotComparable = True

        while catchmentsNotComparable:
            
            #sourceIndex = random.randint(0, 4000)
            sourceIndex = random.randint(0, (len(self.catchments) - 1))

            targetIndex = len(self.catchments) + 1
            randomNum = 0
            
            numTimesTried = 0
            while randomNum == 0 or (targetIndex > (len(self.catchments) - 1)) or (targetIndex < 0) or (numTimesTried < 20):
                randomNum = random.randint(-7, 7)
                targetIndex = sourceIndex + randomNum
                numTimesTried += 1
                #print("here in the little loop")

            sourceCatchment = self.catchments[sourceIndex]
            targetCatchment = self.catchments[targetIndex]
            
            if self.catchmentsComparable(sourceCatchment, targetCatchment):
                catchmentsNotComparable = False
            #print("here")
        return sourceCatchment, targetCatchment


def evaluateModel(model, valChunker, sourceCharacteristics, targetCharacteristics, objective, trainChunker):
    gc.collect()
    model.eval()
    losses = []
    
    sourceCatchmentSize = sourceCharacteristics[1]
    targetCatchmentSize = targetCharacteristics[1]

    preChunk = trainChunker.getPreChunk()

    for _ in range(numChunksPerValidation):

        # get a chunk
        sourceFlowChunk, targetFlowChunk, startIndex = valChunker.getChunk()
        
        predictedFlow = []
        
        fSourceCharacteristics = [float(num) for num in sourceCharacteristics]
        fTargetCharacteristics = [float(num) for num in targetCharacteristics]

        sourceChars = torch.FloatTensor(fSourceCharacteristics)
        targetChars = torch.FloatTensor(fTargetCharacteristics)

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

    predictedFlowChunk = predictedFlowChunk.squeeze().cpu().detach().numpy()
    #predictedFlow = [f * flowCoefficient for f in predictedFlowChunk]

    #print(np.mean(sourceFlow))
    #print(np.mean(targetFlow))
    #print(np.mean(predictedFlowChunk))
    
    np.save("val_ref_sourceFlow",sourceFlow)
    np.save("val_ref_targetFlow",targetFlow)
    np.save("val_ref_preditedFlow", predictedFlowChunk)

    model.train()
    gc.collect()

    return torch.mean(torch.FloatTensor(losses))

def makeTensor(flowString):
    try:
        flowFloat = [float(a) for a in flowString]
        
        #sourceFloat = [float(a) / flowCoefficient for a in sourceString]
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
        #sourceFloat = [float(a) / flowCoefficient for a in sourceString]
    except:
        print('exception in makeFlowTensor:')
        flowFloat = 0.0

    flowTensor = torch.FloatTensor(flowFloat).unsqueeze(dim=0).unsqueeze(dim=0)

    return flowTensor.cuda()


# ************************************************ Action code starts here

# get a list of catchments present in flowFile:
catchments = []

#refCatchmentsPath = "reference_watersheds.csv"
catchmentFile = "corrected_catchment_characteristics.csv"
with open(catchmentFile, "r+") as flowFile:
    i = 0
    for line in flowFile:
        if i > 0:
            lineList = line.split(",")
            catchment = lineList[0]
            if len(catchment) == 7:
                catchment = "0" + catchment
            catchments.append(catchment)
        i = i + 1


picker = CatchmentPicker(catchments, burnFilePath, numYearsToExclude, startYear)

# train!

trainingLosses = []
validationLosses = []

for i in range(last_saved_iteration, numCatchmentsToCompare):

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # cool down the learning rate gradually (the more variability in the data, the lower the lr needs to be) #    if i == 2:
#        optimizer.state_dict()['lr'] = 0.0005
#    if i == 20:
#        optimizer.state_dict()['lr'] = 0.0003
#        numChunksPerComparison = 40
#    if i == 30:
#        optimizer.state_dict()['lr'] = 0.00005        
#        numChunksPerComparison = 50
#        if i == 400:
#            optimizer.state_dict()['lr'] = 0.0001
#        if i == 1000:
#            optimizer.state_dict()['lr'] = 0.00001

    sourceCatchment, targetCatchment = picker.pickTwoCatchments(i)

    sourceCharacteristics = getCatchmentCharacteristics(sourceCatchment)
    targetCharacteristics = getCatchmentCharacteristics(targetCatchment)

    sourceCatchmentSize = sourceCharacteristics[1]
    targetCatchmentSize = targetCharacteristics[1]

    try:
        foo = float(sourceCatchmentSize)
        foo = float(targetCatchmentSize)

        sourceTrainFlow = getFlow(sourceCatchment, 0, trainingDays, picker.burnChecker)
        targetTrainFlow = getFlow(targetCatchment, 0, trainingDays, picker.burnChecker)

        sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, picker.burnChecker)
        targetValFlow = getFlow(targetCatchment, trainingDays, valDays, picker.burnChecker)


        trainChunker = FlowChunker(sourceTrainFlow, targetTrainFlow, chunkLength, sourceCatchmentSize, targetCatchmentSize)
        valChunker = FlowChunker(sourceValFlow, targetValFlow, chunkLength, sourceCatchmentSize, targetCatchmentSize)
   
        fSourceCharacteristics = [float(num) for num in sourceCharacteristics]
        fTargetCharacteristics = [float(num) for num in targetCharacteristics]

        sourceChars = torch.FloatTensor(fSourceCharacteristics).cuda()
        targetChars = torch.FloatTensor(fTargetCharacteristics).cuda()

        print()
        print()
        print("source catchment: " + str(sourceCatchment))
        print("target catchment: " + str(targetCatchment))
        print()

        loop = tqdm(total=numChunksPerComparison, position=0, leave=False)
        
        preChunk = trainChunker.getPreChunk()

        for chunk in range(numChunksPerComparison):
            lr = lr * 0.9995

            optimizer.zero_grad()

            # get a chunk
            sourceFlowChunk, targetFlowChunk, startIndex = trainChunker.getChunk()
            modelInput = np.concatenate((preChunk, sourceFlowChunk), axis=0)
            predictedFlowChunk = model(makeTensor(modelInput).cuda().unsqueeze(dim=0).unsqueeze(dim=0))

            #print(torch.mean(predictedFlowChunk.squeeze()))
            #print(torch.mean(makeFlowTensor(targetFlowChunk).squeeze().squeeze()))
            trainLoss = objective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())
#            trainLoss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 48)
#            trainLoss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 28)
            trainLoss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 24)
            trainLoss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 12) 
            trainLoss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 7) 
            trainLoss += smoothObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze(), 3)
            trainLoss += fourierObjective(predictedFlowChunk.squeeze(), makeTensor(targetFlowChunk).squeeze().squeeze())

            predictedFlowChunk = predictedFlowChunk.squeeze().cpu().detach().numpy()
             
            np.save("ref_sourceFlow",sourceFlowChunk)
            np.save("ref_targetFlow",targetFlowChunk)
            np.save("ref_preditedFlow", predictedFlowChunk)


            if chunk % 5 == 0:
                valLoss = evaluateModel(model, valChunker, sourceCharacteristics, targetCharacteristics, objective, trainChunker)
                validationLosses.append((len(trainingLosses), valLoss))

            trainChunker.update_loss(startIndex, trainLoss.item())

            loop.set_description(
                'comparison_num: {}, trainLoss: {:,.4f}, valLoss: {:,.4f}, num_chunks: {}'.format(i, trainLoss, valLoss,
                                                                                              len(
                                                                                                  list(
                                                                                                      trainChunker.startIndicesToLoss.keys()))))
            loop.update(1)
            trainLoss.backward()
    
            optimizer.step()
    
            trainingLosses.append(trainLoss.item)

        if i % modelSaveInterval == 0:
            torch.save(model.state_dict(), 'big_res_model-' + str(i))
            torch.save(optimizer.state_dict(), 'big_res_optimizer-' + str(i))
            pickle.dump(trainingLosses, open(("big_res_trainingLosses-" + str(i)), "wb"))
            pickle.dump(validationLosses, open(("big_res_validationLosses-" + str(i)), "wb"))
    except:
        print('comparison had problems')

# extract the flow data for each
# remove flow data that is burned or that is reserved for validation and testing ** save the last 20% of valid comparison days (remove anything after Jan 1 2012
#

