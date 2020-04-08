import sys
import random
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
import pickle
import os
import numpy as np

class GruFlowtron(nn.Module):
    def __init__(self, numLayers):
        super(GruFlowtron, self).__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=1000, num_layers=numLayers, bias=True, batch_first=False, dropout=0.1,
                          bidirectional=False)
        self.gruToFlow = nn.Linear(in_features=1000, out_features=1)

    def forward(self, input, hiddens):
        o1, hiddensPrime = self.gru(input.view(1, 1, -1), hiddens)
        o2 = self.gruToFlow(o1)
        return o2, hiddensPrime


# maybe in the future use multiple catchments

catchmentCharacteristicsFilePath = "normalized_zerod_catchment_characteristics.csv"
burnFilePath = "/home/sethbw/Documents/brian_flow_code/Data/catchments_for_Q_analysis.csv"

numYearsToExclude = 10
startYear = 1984

numCatchmentsToCompare = 1000
numChunksPerComparison = 10
numChunksPerValidation = 5
burnIn = 30
chunkLength = 365 + burnIn

trainingDays = 10000 # 10227
valDays = trainingDays + (trainingDays // 10)
testDays = valDays + (trainingDays // 10)
flowCoefficient = 10000  # divide flow by this value to make the value less than 1 in most cases (and multiply the network output by this to get the real flow value!)
numLayers = 8
model = GruFlowtron(numLayers=numLayers).cuda()
model = model
modelSaveInterval = 50
optimizer = torch.optim.Adam(model.parameters())
objective = nn.L1Loss()


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
    #try:
        flowDir = "/home/sethbw/Documents/brian_flow_code/Data/all_flow"
        flowPath = os.path.join(flowDir, (catchmentToGet + ".npy"))
        flow = np.load(flowPath)
        flow.tolist()
        flow = burnChecker.removeBurnedData(flow, catchmentToGet)
        return flow[startIndex:stopIndex]

    #except:
    #    print("exception occured while getting flow")
    #    return []


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


class FlowChunker():

    def __init__(self, sourceCatchmentFlow, targetCatchmentFlow, chunkLength):
        self.chunkLength = chunkLength
        self.startIndicesToLoss = {}
        self.alpha = 40

        self.targetCatchmentFlow = targetCatchmentFlow
        self.sourceCatchmentFlow = sourceCatchmentFlow

    def clear_to_add(self):

        for key in self.startIndicesToLoss.keys():
            if self.startIndicesToLoss[key] > self.alpha:
                return False

        return True

    # check to make sure we have data across all the comparison days
    def verifyIndex(self, startIndex):

        for i in range(self.chunkLength):
            ind = i + startIndex
            try:
                targetFlow = float(self.targetCatchmentFlow[ind])
            except:
                return False
            try:
                sourceFlow = float(self.sourceCatchmentFlow[ind])
            except:
                return False
            # if not targetFlow.isnumeric():
            #     return False
            #
            # if not sourceFlow.isnumeric():
            #     return False
        return True

    def add_new_random_num(self):
        if self.clear_to_add():
            newStartIndex = 0
            newIndexGood = False
            numTimesTried = 0
            numTimesToTry = 10000
            if len(self.startIndicesToLoss.keys()) > 0:
                numTimesToTry = 100
            while not newIndexGood and numTimesTried < numTimesToTry:
                newStartIndex = random.randint(0, (len(self.targetCatchmentFlow) - 1) - self.chunkLength)
                newIndexGood = self.verifyIndex(newStartIndex)
                numTimesTried += 1
            if newStartIndex != 0:
                self.startIndicesToLoss[newStartIndex] = 1000  # initialize to a large number

    def update_loss(self, startIndex, loss):
        self.startIndicesToLoss[startIndex] = loss

    def getChunk(self):

        self.add_new_random_num()
        keys = list(self.startIndicesToLoss.keys())

        if len(keys) == 0:
            self.add_new_random_num()
        if len(keys) == 1:
            knownIndex = 0
        else:
            knownIndex = random.randint(0, len(self.startIndicesToLoss.keys()) - 1)
        startIndex = keys[knownIndex]
        # startIndex = self.start_indices[]
        endIndex = startIndex + self.chunkLength + 1
        return self.targetCatchmentFlow[startIndex:endIndex], self.sourceCatchmentFlow[startIndex:endIndex], startIndex


class CatchmentPicker():
    def __init__(self, catchments, burnFilePath, numYearsToExclude, startYear):
        self.catchments = catchments

        self.burnChecker = BurnChecker(burnFilePath, numYearsToExclude, startYear)

    def flowsComparable(self, sourceFlow, targetFlow):
        comparableLength = 0
        for i in range(len(sourceFlow)):

            if sourceFlow[i].isnumeric() and targetFlow[i].isnumeric():
                comparableLength += 1
            else:
                comparableLength = 0

            if comparableLength > chunkLength:
                return True

        return False

    def catchmentsComparable(self, sourceCatchment, targetCatchment):

        sourceTestFlow = getFlow(sourceCatchment, 0, trainingDays, self.burnChecker)
        sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, self.burnChecker)
        sourceTrainFlow = getFlow(sourceCatchment, valDays, -1, self.burnChecker)

        targetTestFlow = getFlow(targetCatchment, 0, trainingDays, self.burnChecker)
        targetValFlow = getFlow(targetCatchment, trainingDays, valDays, self.burnChecker)
        targetTrainFlow = getFlow(targetCatchment, valDays, -1, self.burnChecker)

        if not self.flowsComparable(sourceTestFlow, targetTestFlow):
            return False
        if not self.flowsComparable(sourceValFlow, targetValFlow):
            return False
        if not self.flowsComparable(sourceTrainFlow, targetTrainFlow):
            return False

        return True

    def pickTwoCatchments(self, comparisonNum):

        catchmentsNotComparable = True

        while catchmentsNotComparable:

            sourceIndex = random.randint(0, (len(self.catchments) // 10))

            # expand number of catchments considered gradually (this helps stabilize the network)
            if comparisonNum > 30:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 2)
            elif comparisonNum > 60:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 3)
            elif comparisonNum > 90:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 4)
            elif comparisonNum > 120:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 5)
            elif comparisonNum > 150:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 6)
            elif comparisonNum > 180:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 7)
            elif comparisonNum > 210:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 8)
            elif comparisonNum > 250:
                 sourceIndex = random.randint(0, (len(self.catchments) // 10) * 9)
            elif comparisonNum > 280:
                sourceIndex = random.randint(0, len(self.catchments) - 1)

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

    # for layer in range(numLayers - 1):
    # hidden = torch.cat((hidden, torch.ones(1000)), dim=0)
    if numLayers > 1:
        newHiddens = []
        newHiddens.append(hidden)
        for layer in range(numLayers - 1):
            newHiddens.append(torch.ones(1000))
            # hidden = torch.cat((hidden, torch.ones(1000)), dim=1)

        hidden = torch.stack(newHiddens)
        hidden = hidden.unsqueeze(dim=1)
    else:
        hidden = hidden.view(1,1,-1)
        # print(hidden.shape)
    # hidden = hidden

    return hidden.cuda()


import matplotlib.pyplot as plt

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

        # for char in chunk:
        output = targetFlowChunk[0]
        predictedFlow.append(output)
        for j in range(1, len(sourceFlowChunk)):
            input = makeFlowTensor(sourceFlowChunk[j], output)
            output, hiddens = model(input, hiddens)
            output = output.squeeze(dim=0).squeeze(dim=0)
            predictedFlow.append(output.item() * flowCoefficient)
            correctOutput = torch.FloatTensor([float(targetFlowChunk[j]) / flowCoefficient]).cuda()
            loss = objective(output, correctOutput)
            if j > burnIn:
                totalLoss = totalLoss + loss

        losses.append(totalLoss)

    # xs = range(chunkLength + 1)
    # print(sourceFlowChunk)
    # print(targetFlowChunk)

    # print(len(xs))
    # print(len(sourceFlowChunk))
    # print(len(targetFlowChunk)) 
    # print(len(predictedFlow))

    sourceFlow = [float(f) for f in sourceFlowChunk[burnIn:]]
    targetFlow = [float(f) for f in targetFlowChunk[burnIn:]]
    predictedFlow = [float(f) for f in predictedFlow[burnIn:]]

    plt.plot(sourceFlow, label="sourceFlow")
    plt.plot(targetFlow, label="targetFlow")
    plt.plot(predictedFlow, label="predictedFlow")
    plt.legend()
    plt.show()

    model.train()
    gc.collect()
    return torch.mean(torch.FloatTensor(losses))


def makeFlowTensor(sourceString, targetString):
    try:
        sourceFloat = float(sourceString) / flowCoefficient
        targetFloat = float(targetString) / flowCoefficient
    except:
        print(sourceFlowChunk)
        print(targetFlowChunk)
        pass

    flowList = [sourceFloat, targetFloat]
    flowTensor = torch.FloatTensor(flowList)

    return flowTensor.cuda()


# ************************************************ Action code starts here

# get a list of catchments present in flowFile:
catchments = []
import os
print(os.getcwd())
referenceCatchmentsPath = "/home/sethbw/Documents/brian_flow_code/Data/reference_watersheds.csv"
with open(referenceCatchmentsPath, "r+") as flowFile:
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

    # cool down the learning rate gradually
    if i == 20:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    if i == 40:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if i == 60:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00009)
    if i == 80:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00008)
        
    if i > 20 and i < 50:
        numChunksPerComparison += 2

    sourceCatchment, targetCatchment = picker.pickTwoCatchments(i)

    sourceTrainFlow = getFlow(sourceCatchment, 0, trainingDays, picker.burnChecker)
    targetTrainFlow = getFlow(targetCatchment, 0, trainingDays, picker.burnChecker)

    sourceValFlow = getFlow(sourceCatchment, trainingDays, valDays, picker.burnChecker)
    targetValFlow = getFlow(targetCatchment, trainingDays, valDays, picker.burnChecker)

    trainChunker = FlowChunker(sourceTrainFlow, targetTrainFlow, chunkLength)
    valChunker = FlowChunker(sourceValFlow, targetValFlow, chunkLength)

    sourceCharacteristics = getCatchmentCharacteristics(sourceCatchment)
    targetCharacteristics = getCatchmentCharacteristics(targetCatchment)
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
        for j in range(1, len(sourceFlowChunk) - 1):
            # get input (the source flow + (target - 1))
            if i > 0:
                input = makeFlowTensor(sourceFlowChunk[j], output)
            else:
                input = makeFlowTensor(sourceFlowChunk[j], targetFlowChunk[j - 1])
            output, hiddens = model(input, hiddens)
            output = output.squeeze(dim=0).squeeze(dim=0)
            predictedFlow.append(output * flowCoefficient)
            correctOutput = torch.FloatTensor([float(targetFlowChunk[j]) / flowCoefficient]).cuda()
            loss = objective(output, correctOutput)
            # if j > burnIn:
            totalLoss = totalLoss + loss

        sourceFlow = [float(f) for f in sourceFlowChunk[burnIn:]]
        targetFlow = [float(f) for f in targetFlowChunk[burnIn:]]
        predictedFlow = [float(f) for f in predictedFlow[burnIn:]]

        predictedFlowTensor = torch.FloatTensor(predictedFlow)

        # clamp the loss at 40
        if i > 20:
            totalLoss = torch.clamp(totalLoss, min=0, max=40)
        if i > 40:
            totalLoss = torch.clamp(totalLoss, min=0, max=30)
        if i > 60:
            totalLoss = torch.clamp(totalLoss, min=0, max=20)
        if i > 80:
            totalLoss = torch.clamp(totalLoss, min=0, max=15)

        # std = predictedFlowTensor.std(dim=0).item()
        # if std < 1000:
        #     print()
        #     print()
        #     lossBonus = (1 / std) * 10000
        #     print('loss bonus')
        #     print(lossBonus)
        #     print()
        #     totalLoss += lossBonus

        # sns.lineplot(y=sourceFlow, x=xs)
        # plt.plot(sourceFlow, label="sourceFlow")
        # plt.plot(targetFlow, label="targetFlow")
        # plt.plot(predictedFlow, label="predictedFlow")
        # plt.legend()
        # plt.show()

        if chunk % 5 == 0:
            valLoss = evaluateModel(model, valChunker, sourceCharacteristics, targetCharacteristics, objective)
            validationLosses.append((len(trainingLosses), valLoss))

        trainChunker.update_loss(startIndex, totalLoss)

        loop.set_description(
            'comparison_num: {}, trainLoss: {:,.4f}, valLoss: {:,.4f}, num_chunks: {}'.format(i, totalLoss, valLoss,
                                                                                              len(
                                                                                                  list(
                                                                                                      trainChunker.startIndicesToLoss.keys()))))

        loop.update(1)
        totalLoss.backward()
        optimizer.step()

        trainingLosses.append(totalLoss.item())

    if i % modelSaveInterval == 0:
        torch.save(model.state_dict(), 'model-' + str(i))
        pickle.dump(trainingLosses, open(("trainingLosses-" + str(i)), "wb"))
        pickle.dump(validationLosses, open(("validationLosses-" + str(i)), "wb"))
# extract the flow data for each
# remove flow data that is burned or that is reserved for validation and testing ** save the last 20% of valid comparison days (remove anything after Jan 1 2012
#

