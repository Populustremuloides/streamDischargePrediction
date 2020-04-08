
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
from tqdm import tqdm


class SummarizeWatershedLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, padding, numLayers = 10, doDropout = False):
        super(SummarizeWatershedLayer, self).__init__()

        self.doDropout = doDropout
        self.dropout = nn.Dropout(0.1)
        self.numLayers = numLayers

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, 
                              dilation=dilation, groups=1, bias=True, padding_mode='zeros') for i in range(numLayers)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(numLayers)])
        self.pls = nn.ModuleList([nn.PReLU(out_channels) for i in range(numLayers)])

    def forward(self, inpt):
        # inpt = self.prl1(self.bn1(self.conv1(input1))) # bring to out_channels
        # oupt = torch.zeros(inpt.shape) # initialize to blank

        for i in range(self.numLayers):
            if self.doDropout:
                oupt = self.pls[i](self.bns[i](self.convs[i](self.dropout(inpt))))
            else:
                oupt = self.pls[i](self.bns[i](self.convs[i](inpt)))
            inpt = oupt.add(inpt)

        return inpt


class CombineWatershedLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, padding, numLayers = 9, doDropout=False):
        super(CombineWatershedLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, 
                              dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prl1 = nn.PReLU(out_channels)

        self.dropout = nn.Dropout(0.1)
        self.doDropout = doDropout
        self.numLayers = numLayers

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, 
                              dilation=dilation, groups=1, bias=True, padding_mode='zeros') for i in range(numLayers)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(numLayers)])
        self.pls = nn.ModuleList([nn.PReLU(out_channels) for i in range(numLayers)])

    def forward(self, inpt):
        inpt = self.prl1(self.bn1(self.conv1(inpt))) # bring to out_channels
        # oupt = torch.zeros(inpt.shape) # initialize to blank

        for i in range(self.numLayers):
            if self.doDropout:
                oupt = self.pls[i](self.bns[i](self.convs[i](self.dropout(inpt))))
            else:
                oupt = self.pls[i](self.bns[i](self.convs[i](inpt)))
            inpt = oupt.add(inpt)

        return inpt




class PredictFlow(nn.Module):

    def __init__(self):
        super(PredictFlow, self).__init__()

        self.layer0 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(1,3),stride=1, padding=(0,1),dilation=(1,1))

        self.layer1 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,1),dilation=(1,1), doDropout=True)
        self.layer2 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,3),dilation=(1,3), doDropout=True)
        self.layer3 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,5),dilation=(1,5), doDropout=True)
        self.layer4 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,7),dilation=(1,7), doDropout=True)
        self.layer5 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,9),dilation=(1,9), doDropout=True)
        self.layer6 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,11),dilation=(1,11), doDropout=True)
        self.layer7 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,1),dilation=(1,1), doDropout=True)
        self.layer8 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,3),dilation=(1,3), doDropout=True)
        self.layer9 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,3),dilation=(1,3), doDropout=True)
        self.layer10 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(1,3),stride=1, padding=(0,1),dilation=(1,1), doDropout=True)

        # begin convolving watersheds together
        self.layer11 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,1),dilation=(1,1), doDropout=True)
        self.layer12 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,3),dilation=(1,3), doDropout=True)
        self.layer13 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,5),dilation=(1,5), doDropout=True)
        self.layer14 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,7),dilation=(1,7), doDropout=True)
        self.layer15 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,9),dilation=(1,9), doDropout=True)
        self.layer16 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,11),dilation=(1,11), doDropout=True)
        self.layer17 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,3),dilation=(1,3), doDropout=True)
        self.layer18 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,3),dilation=(1,3))
        self.layer19 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,1),dilation=(1,1))
        self.layer20 = SummarizeWatershedLayer(in_channels=100,out_channels=100,kernel_size=(3,3),stride=1, padding=(1,1),dilation=(1,1))

        # condense
        self.pool21 = nn.AvgPool2d((2,2))
        self.layer21 = CombineWatershedLayer(in_channels=100,out_channels=50,kernel_size=(3,3),stride=1, padding=(1,1),dilation=(1,1))
        self.pool22 = nn.AvgPool2d((1,2), stride=2)
        self.layer22 = CombineWatershedLayer(in_channels=50,out_channels=25,kernel_size=(3,3),stride=1, padding=(1,1),dilation=(1,1))
        self.pool23 = nn.AvgPool2d((1,2), stride=2)
        self.layer23 = CombineWatershedLayer(in_channels=25,out_channels=5,kernel_size=(3,3),stride=1, padding=(1,1),dilation=(1,1))
        self.pool24 = nn.AvgPool2d((1,2), stride=2)
        self.layer24 = CombineWatershedLayer(in_channels=5,out_channels=1,kernel_size=(3,3),stride=1, padding=(1,1),dilation=(1,1))

        self.layer25 = nn.Linear(in_features=30, out_features=1)

    def forward(self, x):
        b0 = self.layer0(x)
        b1 = self.layer1(b0)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        b5 = self.layer5(b4)
        b6 = self.layer6(b5)
        b7 = self.layer7(b6)
        b8 = self.layer8(b7)
        b9 = self.layer9(b8)
        b10 = self.layer10(b9)

        # summarize the watersheds together

        b11  = self.layer11(b10)
        b12  = self.layer12(b11)
        b13  = self.layer13(b12)
        b14  = self.layer14(b13)
        b15  = self.layer15(b14)
        b16  = self.layer16(b15)
        b17  = self.layer17(b16)
        b18  = self.layer18(b17)
        b19  = self.layer19(b18)
        b20  = self.layer20(b19)

        # condense output
        b21  = self.layer21(self.pool21(b20))
        b22  = self.layer22(self.pool22(b21))
        b23  = self.layer23(self.pool23(b22))
        b24  = self.layer24(self.pool24(b23))

        # final, linear layer
        b25  = self.layer25(b24)

        return b25.view(-1)
