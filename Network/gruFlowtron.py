import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
from tqdm import tqdm
class GruFlowtron(nn.Module):

    def __init__(self,):

        super(GruFlowtron, self).__init__()         
        
        self.rnn = nn.GRU()

    def forward(self, input, hiddens): 
        return 

