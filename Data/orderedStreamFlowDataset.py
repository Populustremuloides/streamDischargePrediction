import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import os
import pickle


# get a chunk of time
# then break that time up into individual pieces


class USStreamFlowDataset(Dataset):
    def __init__(self, root_dir, year, region):

        # can I modify this to handle multiple regions?

        self.rootDir = root_dir
        
        self.filePrefix = str(year) + "_" + str(region) + "_"
        self.region = str(region)
        # get the indices of the train, test, and validation files

        trainIndicesFile = "trainIndices"
        testIndicesFile = "testIndices"
        valIndicesFile = "validationIndices"
        
        trainPath = os.path.join(self.rootDir, trainIndicesFile)
        testPath = os.path.join(self.rootDir, testIndicesFile)
        valPath = os.path.join(self.rootDir, valIndicesFile)

        self.trainIndices =pickle.load(open(trainPath, "rb"))
        self.testIndices =pickle.load(open(testPath, "rb"))
        self.valIndices =pickle.load(open(valPath, "rb"))

        self.length = (len(self.trainIndices) + 
                       len(self.testIndices)  + 
                       len(self.valIndices)    )


    def __len__(self):
        return self.length

    
    def __getitem__(self, index): # maybe add a region thing here, too?
        fileName = self.region + "_" + str(index)
        filePath = os.path.join(self.rootDir, fileName)
        
        item =pickle.load(open(filePath, "rb"))
        x = torch.FloatTensor(item[0]).unsqueeze(dim=0)
        y = torch.FloatTensor([item[1]])

        return x, y




