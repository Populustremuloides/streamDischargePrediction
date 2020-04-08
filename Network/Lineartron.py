
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
#!pip3 install tqdm
from tqdm import tqdm

class LinearTron(nn.Module):

    def __init__(self,):

        super(LinearTron, self).__init__() # allows you to access nn.Module as a parent class
        
        self.double1 = nn.Linear(in_features = 950, out_features=1)

    def forward(self, input): 

        return self.double1(input)
        
