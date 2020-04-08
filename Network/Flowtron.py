
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
from tqdm import tqdm

class Flowtron(nn.Module):

    def __init__(self,):

        super(Flowtron, self).__init__() # allows you to access nn.Module as a parent class
        
        self.dropout = nn.Dropout(p=0.25)

        # block 1
        self.double1 = nn.Linear(in_features = 962, out_features=512)
        self.d1_PReLU = nn.PReLU(512)
        self.single1 = nn.Linear(in_features = 512, out_features=512)
        self.s1_PReLU = nn.PReLU(512)
        self.batchNorm1 = nn.BatchNorm1d(512)

        # block 2
        self.double2 = nn.Linear(in_features = 512, out_features=512)
        self.d2_PReLU = nn.PReLU(512)
        self.single2 = nn.Linear(in_features = 512, out_features=512)
        self.s2_PReLU = nn.PReLU(512)
        self.batchNorm2 = nn.BatchNorm1d(512)

        # block 3
        self.double3 = nn.Linear(in_features = 512, out_features=512)
        self.d3_PReLU = nn.PReLU(512)
        self.single3 = nn.Linear(in_features = 512, out_features=512)
        self.s3_PReLU = nn.PReLU(512)
        self.batchNorm3 = nn.BatchNorm1d(512)

        # block 4
        self.double4 = nn.Linear(in_features = 512, out_features=512)
        self.d4_PReLU = nn.PReLU(512)
        self.single4 = nn.Linear(in_features = 512, out_features=512)
        self.s4_PReLU = nn.PReLU(512)
        self.batchNorm4 = nn.BatchNorm1d(512)

        # block 5
        self.double5 = nn.Linear(in_features = 512, out_features=512)
        self.d5_PReLU = nn.PReLU(512)
        self.single5 = nn.Linear(in_features = 512, out_features=512)
        self.s5_PReLU = nn.PReLU(512)
        self.batchNorm5 = nn.BatchNorm1d(512)

        # block 6
        self.double6 = nn.Linear(in_features = 512, out_features=512)
        self.d6_PReLU = nn.PReLU(512)
        self.single6 = nn.Linear(in_features = 512, out_features=512)
        self.s6_PReLU = nn.PReLU(512)
        self.batchNorm6 = nn.BatchNorm1d(512)

        # block 7
        self.double7 = nn.Linear(in_features = 512, out_features=512)
        self.d7_PReLU = nn.PReLU(512)
        self.single7 = nn.Linear(in_features = 512, out_features=512)
        self.s7_PReLU = nn.PReLU(512)
        self.batchNorm7 = nn.BatchNorm1d(512)

        # block 8
        self.double8 = nn.Linear(in_features = 512, out_features=512)
        self.d8_PReLU = nn.PReLU(512)
        self.single8 = nn.Linear(in_features = 512, out_features=512)
        self.s8_PReLU = nn.PReLU(512)
        self.batchNorm8 = nn.BatchNorm1d(512)

        # block 9
        self.double9 = nn.Linear(in_features = 512, out_features=512)
        self.d9_PReLU = nn.PReLU(512)
        self.single9 = nn.Linear(in_features = 512, out_features=512)
        self.s9_PReLU = nn.PReLU(512)
        self.batchNorm9 = nn.BatchNorm1d(512)

        # block 10
        self.double10 = nn.Linear(in_features = 512, out_features=256)
        self.d10_PReLU = nn.PReLU(256)
        self.single10 = nn.Linear(in_features = 256, out_features=128)
        self.s10_PReLU = nn.PReLU(128)

        # block 11
        self.double11 = nn.Linear(in_features = 128, out_features=64)
        self.d11_PReLU = nn.PReLU(64)
        self.single11 = nn.Linear(in_features = 64, out_features=32)
        self.s11_PReLU = nn.PReLU(32)

        # block 12
        self.double12 = nn.Linear(in_features = 32, out_features=16)
        self.d12_PReLU = nn.PReLU(16)
        self.single12 = nn.Linear(in_features = 16, out_features=1)
        self.s12_PReLU = nn.PReLU(1)

    def forward(self, input): 
        
        out1_1 = self.d1_PReLU(self.double1(input))
        out1_2 = self.batchNorm1(self.s1_PReLU(self.single1(out1_1)))

        out2_1 = self.d2_PReLU(self.double2(self.dropout(out1_2)))
        out2_2 = self.batchNorm2(self.s2_PReLU(self.single2(self.dropout(out2_1))))

        out3_1 = self.d3_PReLU(self.double3(self.dropout(out2_2)))
        out3_2 = self.batchNorm3(self.s3_PReLU(self.single3(self.dropout(out3_1))))

        out4_1 = self.d4_PReLU(self.double4(self.dropout(out3_2)))
        out4_2 = self.batchNorm4(self.s4_PReLU(self.single4(self.dropout(out4_1))))

        out5_1 = self.d5_PReLU(self.double5(self.dropout(out4_2)))
        out5_2 = self.batchNorm5(self.s5_PReLU(self.single5(self.dropout(out5_1))))

        out6_1 = self.d6_PReLU(self.double6(self.dropout(out5_2)))
        out6_2 = self.batchNorm6(self.s6_PReLU(self.single6(self.dropout(out6_1))))

        out7_1 = self.d7_PReLU(self.double7(self.dropout(out6_2)))
        out7_2 = self.batchNorm7(self.s7_PReLU(self.single7(self.dropout(out7_1))))

        out8_1 = self.d8_PReLU(self.double8(self.dropout(out7_2)))
        out8_2 = self.batchNorm8(self.s8_PReLU(self.single8(self.dropout(out8_1))))

        out9_1 = self.d9_PReLU(self.double9(self.dropout(out8_2)))
        out9_2 = self.batchNorm9(self.s9_PReLU(self.single9(self.dropout(out9_1))))

        out10_1 = self.d10_PReLU(self.double10(self.dropout(out9_2)))
        out10_2 = self.s10_PReLU(self.single10(self.dropout(out10_1)))

        out11_1 = self.d11_PReLU(self.double11(self.dropout(out10_2)))
        out11_2 = self.s11_PReLU(self.single11(self.dropout(out11_1)))

        out12_1 = self.d12_PReLU(self.double12(out11_2))
        out12 = self.s12_PReLU(self.single12(out12_1))

        return out12

