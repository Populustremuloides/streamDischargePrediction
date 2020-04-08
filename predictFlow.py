import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from Network.ResFlowtron import *
from Network.Flowtron import *
from Network.PredictFlow import *
from Network.Lineartron import *
from Data.USStreamFlowDataset import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# user altered variables
root = "/home/sethbw/Documents/brian_flow_code/Data/1952_05_data"
modelFolder = "/home/sethbw/Documents/brian_flow_code/Models/1952_05"
dataset = USStreamFlowDataset(root, "1952", "05")


# global variables
batchSize = 1
numEpochs = 5

guidedTime = 365




# I am going to do this with the original data
# if I am going to do this, I need a way to get data sequentially
def updateXNext(yHat, xNext):
    # I need to make the y part of xNext change according to yHat
    previousFlow = xNext[0,-14:-2]
    newDay = yHat
    newPreviousFlow = previousFlow + newDay
    xNext[0:-14:-1] = newPreviousFlow

    # 0 is the source
    # 1 is the target
    x[-1] = yHat
    # shift the last 13 left by one

    return xNext


#criterion = nn.L1Loss()


# cuda = torch.cuda.is_available()
cuda = True
model = PredictFlow()

if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

optimizer = optim.Adam(model.parameters())

# get the data

trainLosses = []
valLosses = []

def shuffleDataset(dataset):

    trainIndices = dataset.trainIndices
    valIndices = dataset.valIndices

    np.random.shuffle(trainIndices)
    np.random.shuffle(valIndices)
   
    trainSampler = SubsetRandomSampler(dataset.trainIndices)
    valSampler = SubsetRandomSampler(dataset.valIndices)

    trainLoader = torch.utils.data.DataLoader(dataset, 
        batch_size=batchSize, sampler=trainSampler)
    valLoader = torch.utils.data.DataLoader(dataset,
        batch_size=batchSize, sampler=trainSampler)

    return trainLoader, valLoader


# get the model

for epoch in range(numEpochs):
    print("new epoch")

    trainLoader, valLoader = shuffleDataset(dataset)
    numBatches = len(dataset.trainIndices) // batchSize
    print("len(dataset.trainIndices:" + str(len(dataset.trainIndices)))
    print("batchSize: " + str(batchSize))
    loop = tqdm(numBatches, position=0, leave=False)

    for batchIndex, (x, y) in enumerate(trainLoader):
        model.train()
        model.zero_grad()

        if cuda:
            x = x.cuda()
            y = y.cuda()
        
        yHat = model(x)

        print(yHat)

        batchLoss = criterion(yHat, y)

        trainLosses.append(batchLoss.item())

        batchLoss.backward()

        optimizer.step()


        if batchIndex % evalInterval == 0:
            gc.collect()
            model.eval()
            valX, valY = iter(valLoader).next()

            if cuda:
                valX = valX.cuda()
                valY = valY.cuda()

            valYHat = model(valX)
            valLoss = criterion(valYHat, valY)
            valLoss = valLoss.item()

            valLosses.append((len(trainLosses), valLoss))

            model.train()
            gc.collect()


        if batchIndex % saveInterval == 0:
            epochFolder = os.path.join(modelFolder, "epoch_" + str(epoch))
            if os.path.exists(epochFolder):
                pass
            else:
                os.makedirs(epochFolder)
            torch.save(model.state_dict(),
                    os.path.join(epochFolder, "model-" + str(batchIndex)))
           
            valLossFile = os.path.join(epochFolder, (str(batchIndex) + "validationLosses.p"))
            trainLossFile = os.path.join(epochFolder, (str(batchIndex) + "trainLosses.p"))

            pickle.dump(valLosses, open(valLossFile, "wb"))
            pickle.dump(trainLosses, open(trainLossFile, "wb"))

                
        memoryUsed = str(torch.cuda.memory_allocated(0) / 100000)

        loop.set_description("batch: {}, loss: {}, valLoss: {}, memory: {}, totalBatches: {}".format(
            str(batchIndex), str(batchLoss.item()), str(valLoss), memoryUsed, str(numBatches)))
        loop.update(1)


