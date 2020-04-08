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

class GruFlowtron(nn.Module):
    def __init__(self, numLayers):
        super(GruFlowtron, self).__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=1000, num_layers=numLayers, bias=True, batch_first=False, dropout=0.1,
                          bidirectional=False)
        self.gruToFlow = nn.Linear(in_features=1000, out_features=1)

    def forward(self, input, hiddens):
        o1, hiddensPrime = self.gru(input.view(1, 1, -1), hiddens)

        flow = self.gruToFlow(o1)

        return flow, hiddensPrime


# maybe in the future use multiple catchments

catchmentCharacteristicsFilePath = "normalized_zerod_catchment_characteristics.csv"
burnFilePath = "catchments_for_Q_analysis.csv"

outputFile = "test_output.csv"

with open(outputFile, "w+") as oFile:
    oFile.write("testNum,dataType,correlationToTarget,catchmentId,FlowValues\n")

numYearsToExclude = 10
startYear = 1972

numCatchmentsToCompare = 500
numChunksPerComparison = 200
numChunksPerValidation = 5
numChunksPerTest = 1
trainBurnIn = 10
burnIn = 0 + trainBurnIn
chunkLength = 365 + burnIn

trainingDays = 10000 # 10227
valDays = trainingDays + (trainingDays // 10)
testDays = valDays + (trainingDays // 10)
flowCoefficient = 10000  # divide flow by this value to make the value less than 1 in most cases (and multiply the network output by this to get the real flow value!)
numLayers = 8
model = GruFlowtron(numLayers=numLayers)
stateDict = torch.load("model-19")
model.load_state_dict(stateDict)
model.cuda()

numLosses = 0
numWins = 0

for parameter in model.parameters():
    parameter.requires_grad = True

#modelSaveInterval = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
optimizer.load_state_dict(torch.load('optimizer-19'))
optimizer.state_dict()['lr'] = 0.00005

numTestsRun = 0

class PercentageLoss(nn.Module):
    def __init__(self,):
        super(PercentageLoss, self).__init__()

    def forward(self, yHat, y):
        diff = (torch.abs(yHat - y)) + (torch.abs(yHat - y) ** 2)
        percentDiff = diff / y
        return torch.mean(percentDiff) / y.shape[0]

#class getCorrelation(nn.Module):
#    def __init__(self,):
#        super(CorrelationLoss, self).__init__()

#    def forward(self, yHat, y):
#        print("in forward")
#        print(len(yHat))
#        print(len(y))

#        correlation = pearsonr(yHat, y)
#        correlation = correlation[0] * 100
#        loss = [100 - correlation]
#        return torch.FloatTensor(loss)

objective = nn.L1Loss() # CorrelationLoss()  #PercentageLoss() # nn.MSELoss() #nn.L1Loss() # PercentageLoss() #  nn.L1Loss()


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
                        flow[index] = None
        return flow


class FlowChunker():

    def __init__(self, sourceCatchmentFlow, targetCatchmentFlow, chunkLength, alpha):

        self.chunkLength = chunkLength
        self.startIndicesToLoss = {}
        self.alpha = alpha

        self.targetCatchmentFlow = targetCatchmentFlow
        self.sourceCatchmentFlow = sourceCatchmentFlow

        self.comparableIndices = self.getComparableIndices()

        # add the first chunk
        newKey = self.comparableIndices[0]
        self.startIndicesToLoss[newKey] = int(1000)

    def getComparableIndices(self):
        comparableIndices = []

        minLength = min(len(self.sourceCatchmentFlow),len(self.targetCatchmentFlow))
        indicesToCheck = minLength - chunkLength - 19 
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
        self.startIndicesToLoss[startIndex] = loss.item()

    def getChunk(self):

        self.add_new_random_num()
        keys = list(self.startIndicesToLoss.keys())
        randomStartIndex = random.randint(0, len(self.startIndicesToLoss.keys()) - 1)
        
        startIndex = keys[randomStartIndex]
        endIndex = startIndex + self.chunkLength

        targetFlowSelection = self.targetCatchmentFlow[startIndex:endIndex]
        sourceFlowSelection = self.sourceCatchmentFlow[startIndex:endIndex]
        
        return targetFlowSelection, sourceFlowSelection, startIndex


class CatchmentPicker():
    def __init__(self, catchments, burnFilePath, numYearsToExclude, startYear):
        self.catchments = catchments

        self.burnChecker = BurnChecker(burnFilePath, numYearsToExclude, startYear)

    def flowsComparable(self, sourceCatchmentFlow, targetCatchmentFlow):

        minLength = min(len(sourceCatchmentFlow),len(targetCatchmentFlow))
        indicesToCheck = minLength - chunkLength - 19 
        if indicesToCheck < 0:
            indicesToCheck = 0

        for i in range(indicesToCheck): # avoid the edge, where there is likely to be error
           
            numContinuousValidDays = 0
            # examine every part of the chunk length to verify it is non-zero
            for j in range(chunkLength):
                if not (sourceCatchmentFlow[i + j] == None) and not (targetCatchmentFlow[i + j] == None):
                    numContinuousValidDays += 1
     
            if numContinuousValidDays == chunkLength:
                try:
                    correlation = pearsonr(sourceCatchmentFlow[i:i + chunkLength], targetCatchmentFlow[i:i + chunkLength])
                except:
                    print("***********************************************************************************")
                    print("***********************************************************************************")
                    print("***********************************************************************************")
                    print("***********************************************************************************")
                    print(sourceCatchmentFlow[i:i + chunkLength])
                    print(targetCatchmentFlow[i:i + chunkLength])

                if correlation[0] > 0.75:
                    print("winning correlation! " + str(correlation[0]))
                    return True
                else:
                    print('correlation was: ' + str(correlation[0]))
                    return False
 
        return False
        

    def catchmentsComparable(self, sourceCatchment, targetCatchment):

        sourceTestFlow = getFlow(sourceCatchment, 0, trainingDays, self.burnChecker)
        sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, self.burnChecker)
        sourceTrainFlow = getFlow(sourceCatchment, valDays, testDays, self.burnChecker)

        targetTestFlow = getFlow(targetCatchment, 0, trainingDays, self.burnChecker)
        targetValFlow = getFlow(targetCatchment, trainingDays, valDays, self.burnChecker)
        targetTrainFlow = getFlow(targetCatchment, valDays, testDays, self.burnChecker)

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

            sourceIndex = random.randint(0, (len(self.catchments) - 1))

            targetIndex = len(self.catchments) + 1
            randomNum = 0
            while randomNum == 0 or (targetIndex > (len(self.catchments) - 1)):
                randomNum = random.randint(-5, 5)

                targetIndex = sourceIndex + randomNum

            sourceCatchment = self.catchments[sourceIndex]
            targetCatchment = self.catchments[targetIndex]
            
            if self.catchmentsComparable(sourceCatchment, targetCatchment):
                catchmentsNotComparable = False

        return sourceCatchment, targetCatchment


def initializeHiddens(sourceCatchmentChars, targetCatchmentChars, numLayers):
    try:
        zerosSource = torch.ones(32)
        source = [float(i) for i in sourceCatchmentChars[1:]]
        source = torch.FloatTensor(source)
    except:
        print('source')
        print(sourceCatchmentChars)
        return -1

    try:
        zerosTarget = torch.ones(32)
        target = [float(i) for i in targetCatchmentChars[1:]]
        target = torch.FloatTensor(target)
    except:
        print('target')
        print(targetCatchmentChars)
        return -1
        pass

    hidden = torch.cat((zerosSource, source, zerosTarget, target), dim=0)
    
#    print(hidden.shape)
    # for layer in range(numLayers - 1):
    # hidden = torch.cat((hidden, torch.ones(1000)), dim=0)
    if numLayers > 1:
        newHiddens = []
#        newHiddens.append(torch.ones(1000))
        newHiddens.append(hidden)
        for layer in range(numLayers - 1):
#            newHiddens.append(hidden)
            newHiddens.append(torch.ones(1000))
            # hidden = torch.cat((hidden, torch.ones(1000)), dim=1)

        hidden = torch.stack(newHiddens)
        hidden = hidden.unsqueeze(dim=1)
    else:
        hidden = hidden.view(1,1,-1)
        # print(hidden.shape)
    # hidden = hidden

    return hidden.cuda()



def evaluateModel(model, valChunker, sourceCharacteristics, targetCharacteristics, objective):
    gc.collect()
    model.eval()
    losses = []
    for _ in range(numChunksPerValidation):

        totalLoss = 0

        # get a chunk
        sourceFlowChunk, targetFlowChunk, startIndex = valChunker.getChunk()

        predictedFlow = []
        # initiate the hiddens
        hiddens = initializeHiddens(sourceCharacteristics, targetCharacteristics, numLayers)
        if type(hiddens) == type(-1):
            print('eval')
            print(sourceCharacteristics)
            print(targetCharacteristics)

        fSourceCharacteristics = [float(num) for num in sourceCharacteristics]
        fTargetCharacteristics = [float(num) for num in targetCharacteristics]

        sourceChars = torch.FloatTensor(fSourceCharacteristics)
        targetChars = torch.FloatTensor(fTargetCharacteristics)
        catchmentInfo = torch.cat((sourceChars, targetChars), dim=0).cuda()

        # for char in chunk:
        output = targetFlowChunk[0]
        predictedFlow.append(output)
        for j in range(1, len(sourceFlowChunk)):
            input = makeFlowTensor(sourceFlowChunk[j], output)
#            input = torch.cat((input, catchmentInfo), dim=0)

            output, hiddens = model(input, hiddens)
            output = output.squeeze(dim=0).squeeze(dim=0)
            predictedFlow.append(output.item() * flowCoefficient)
            correctOutput = torch.FloatTensor([float(targetFlowChunk[j]) / flowCoefficient]).cuda()
            loss = objective(output, correctOutput)
            if j > trainBurnIn:
                totalLoss = totalLoss + loss
        losses.append(totalLoss)

    # xs = range(chunkLength + 1)
    # print(sourceFlowChunk)
    # print(targetFlowChunk)

    # print(len(xs))
    # print(len(sourceFlowChunk))
    # print(len(targetFlowChunk)) 
    # print(len(predictedFlow))

    sourceFlow = [float(f) for f in sourceFlowChunk]
    targetFlow = [float(f) for f in targetFlowChunk]
    predictedFlow = [float(f) for f in predictedFlow]

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




def testModel(model, testChunker, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, objective, outputFile):
    gc.collect()
    model.eval()
    losses = []
    for _ in range(numChunksPerTest):

        totalLoss = 0

        # get a chunk
        sourceFlowChunk, targetFlowChunk, startIndex = testChunker.getChunk()

        predictedFlow = []
        # initiate the hiddens
        hiddens = initializeHiddens(sourceCharacteristics, targetCharacteristics, numLayers)
        if type(hiddens) == type(-1):
            print('eval')
            print(sourceCharacteristics)
            print(targetCharacteristics)

        fSourceCharacteristics = [float(num) for num in sourceCharacteristics]
        fTargetCharacteristics = [float(num) for num in targetCharacteristics]

        sourceChars = torch.FloatTensor(fSourceCharacteristics)
        targetChars = torch.FloatTensor(fTargetCharacteristics)
        catchmentInfo = torch.cat((sourceChars, targetChars), dim=0).cuda()

        # for char in chunk:
        output = targetFlowChunk[0]
        predictedFlow.append(output)
        for j in range(1, len(sourceFlowChunk)):
            input = makeFlowTensor(sourceFlowChunk[j], output)
#            input = torch.cat((input, catchmentInfo), dim=0)

            output, hiddens = model(input, hiddens)
            output = output.squeeze(dim=0).squeeze(dim=0)
            predictedFlow.append(output.item() * flowCoefficient)
            correctOutput = torch.FloatTensor([float(targetFlowChunk[j]) / flowCoefficient]).cuda()
            loss = objective(output, correctOutput)
            if j > trainBurnIn:
                totalLoss = totalLoss + loss

            losses.append(totalLoss)
        
        # remove anything less than 0

        sourceFlow = [max(f,0) for f in sourceFlowChunk]
        targetFlow = [max(f,0) for f in targetFlowChunk]
        predictedFlow = [max(f,0) for f in predictedFlow]
        
        # calculate the correlations
        predCorrelation = pearsonr(targetFlow, predictedFlow)
        predCorrelation = predCorrelation[0]
        realCorrelation = pearsonr(targetFlow, sourceFlow)
        realCorrelation = realCorrelation[0]

        # record the information

        sourceFlow = [str(f) for f in sourceFlowChunk]
        targetFlow = [str(f) for f in targetFlowChunk]
        predictedFlow = [str(f) for f in predictedFlow]


        with open(outputFile, "a") as oFile:
            sourceLine = [str(numTestsRun),"source", str(realCorrelation), str(sourceCatchment)]
            sourceLine = sourceLine + sourceFlow
            sourceLine = ",".join(sourceLine)

            targetLine = [str(numTestsRun),"target", "1", str(targetCatchment)]
            targetLine = targetLine + targetFlow
            targetLine = ",".join(targetLine)

            predictedLine = [str(numTestsRun),"predicted", str(predCorrelation), str(targetCatchment)]
            predictedLine = predictedLine + predictedFlow
            predictedLine = ",".join(predictedLine)     

            oFile.write(sourceLine  + '\n')
            oFile.write(predictedLine + '\n')
            oFile.write(targetLine + '\n')

    model.train()
    gc.collect()
    if predCorrelation > realCorrelation:
        return True
    else:
        return False




def makeFlowTensor(sourceString, targetString):
    try:
        sourceFloat = float(sourceString) / flowCoefficient
    except:
        print('exception in makeFlowTensor:')
        sourceFloat = 0.0
    try:
        targetFloat = float(targetString) / flowCoefficient
    except:
        print('exception in makeFlowTensor:')
        targetFloat = 0.0

    flowList = [sourceFloat, targetFloat] # * 1000
    flowTensor = torch.FloatTensor(flowList)

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
            catchments.append(catchment)
        i = i + 1


picker = CatchmentPicker(catchments, burnFilePath, numYearsToExclude, startYear)

# train!

trainingLosses = []
validationLosses = []

for i in range(numCatchmentsToCompare):
    
    # reset the model
    model = GruFlowtron(numLayers=numLayers)
    stateDict = torch.load("model-19")
    model.load_state_dict(stateDict)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    optimizer.load_state_dict(torch.load('optimizer-19'))
    optimizer.state_dict()['lr'] = 0.00005
    
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.train()


    # cool down the learning rate gradually (the more variability in the data, the lower the lr needs to be)
#    if i > 19 and i < 50:
#        numChunksPerComparison += 2

    sourceCatchment, targetCatchment = picker.pickTwoCatchments(i)

    sourceTrainFlow = getFlow(sourceCatchment, 0, trainingDays, picker.burnChecker)
    targetTrainFlow = getFlow(targetCatchment, 0, trainingDays, picker.burnChecker)

    sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, picker.burnChecker)
    targetValFlow = getFlow(targetCatchment, trainingDays, valDays, picker.burnChecker)

    sourceTestFlow = getFlow(sourceCatchment, valDays, testDays, picker.burnChecker)
    targetTestFlow = getFlow(targetCatchment, valDays, testDays, picker.burnChecker)

    trainChunker = FlowChunker(sourceTrainFlow, targetTrainFlow, chunkLength, 500)
    valChunker = FlowChunker(sourceValFlow, targetValFlow, chunkLength, 1000)
    testChunker = FlowChunker(sourceTestFlow, targetTestFlow, chunkLength, 1000)

    sourceCharacteristics = getCatchmentCharacteristics(sourceCatchment)
    targetCharacteristics = getCatchmentCharacteristics(targetCatchment)

    fSourceCharacteristics = [float(num) for num in sourceCharacteristics]
    fTargetCharacteristics = [float(num) for num in targetCharacteristics]

    sourceChars = torch.FloatTensor(fSourceCharacteristics)
    targetChars = torch.FloatTensor(fTargetCharacteristics)
    catchmentInfo = torch.cat((sourceChars, targetChars), dim=0).cuda()

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
        hiddens = initializeHiddens(sourceCharacteristics, targetCharacteristics, numLayers)
        if type(hiddens) == type(-1):
            print('train')
            print(sourceCatchment)
            print(sourceCharacteristics)
            print(targetCatchment)
            print(targetCharacteristics)
        # for char in chunk:
        output = targetFlowChunk[0]
        predictedFlow = []
        predictedFlow.append(output)
        for j in range(1, len(sourceFlowChunk)):
            # get input (the source flow + (target - 1))
            if i > 0:
                input = makeFlowTensor(sourceFlowChunk[j], output)
#                input = torch.cat((input,catchmentInfo), dim=0)
            else:
                input = makeFlowTensor(sourceFlowChunk[j], targetFlowChunk[j - 1])
#                input = torch.cat((input,catchmentInfo), dim=0)


            output, hiddens = model(input, hiddens)
            output = output.squeeze(dim=0).squeeze(dim=0)
            predictedFlow.append(output * flowCoefficient)
            correctOutput = torch.FloatTensor([float(targetFlowChunk[j]) / flowCoefficient]).cuda()
            loss = objective(output, correctOutput)

            if j > trainBurnIn:
                totalLoss = totalLoss + loss

        if chunk % 5 == 0:
            valLoss = evaluateModel(model, valChunker, sourceCharacteristics, targetCharacteristics, objective)
            validationLosses.append((len(trainingLosses), valLoss))

        trainChunker.update_loss(startIndex, totalLoss)
        numChunks = str(len(trainChunker.startIndicesToLoss.keys()))
        loop.set_description(
            'comparison_num: {}, trainLoss: {:,.4f}, valLoss: {:,.4f}, num_chunks: {}'.format(i, totalLoss.item(), valLoss,numChunks))

        if chunk == (numChunksPerComparison - 1):
            # test the accuracy on the 
            won = testModel(model, testChunker, sourceCharacteristics, targetCharacteristics, sourceCatchment, targetCatchment, objective, outputFile)
            if won:
                numWins += 1
                print("WON!")
            else:
                numLosses += 1
                print("lost")

        loop.update(1)
        totalLoss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        trainingLosses.append(totalLoss.item())


print("number of wins: " + str(numWins))
print("number of losses: " + str(numLosses))

#    if i % modelSaveInterval == 0:
#        torch.save(model.state_dict(), 'model-' + str(i))
#        pickle.dump(trainingLosses, open(("trainingLosses-" + str(i)), "wb"))
#        pickle.dump(validationLosses, open(("validationLosses-" + str(i)), "wb"))
# extract the flow data for each
# remove flow data that is burned or that is reserved for validation and testing ** save the last 19% of valid comparison days (remove anything after Jan 1 1912
#

