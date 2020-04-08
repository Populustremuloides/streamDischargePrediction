import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
from tqdm import tqdm
class ResFlowtron(nn.Module):

    def __init__(self,):

        super(ResFlowtron, self).__init__()         

        self.dropout = nn.Dropout(p=0.2)


        # block 1
        self.double1 = nn.Linear(in_features = 950, out_features=1024)
        self.bn1_0 = nn.BatchNorm1d(1024)
        self.d1_PReLU = nn.PReLU(1024)

        self.single1_1 = nn.Linear(in_features = 1024, out_features = 512)
        self.bn1_1 = nn.BatchNorm1d(512)
        self.s11_PReLU = nn.PReLU(512)

        self.single1_2 = nn.Linear(in_features = 512, out_features = 256)
        self.bn1_2 = nn.BatchNorm1d(256)
        self.s12_PReLU = nn.PReLU(256)

        self.single1_3 = nn.Linear(in_features = 256, out_features=128)
        self.bn1_3 = nn.BatchNorm1d(128)
        self.s13_PReLU = nn.PReLU(128)

        # block 2
        self.double2 = nn.Linear(in_features = 1024 + 128, out_features=1024) 
        self.bn2_0 = nn.BatchNorm1d(1024)
        self.d2_PReLU = nn.PReLU(1024)
 
        self.single2_1 = nn.Linear(in_features = 1024, out_features = 512)
        self.bn2_1 = nn.BatchNorm1d(512)
        self.s21_PReLU = nn.PReLU(512)

        self.single2_2 = nn.Linear(in_features = 512, out_features = 256)
        self.bn2_2 = nn.BatchNorm1d(256)
        self.s22_PReLU = nn.PReLU(256)

        self.single2_3 = nn.Linear(in_features = 256, out_features=128)
        self.bn2_3 = nn.BatchNorm1d(128)
        self.s23_PReLU = nn.PReLU(128)

        # block 3
        self.double3 = nn.Linear(in_features = 1024 + 128, out_features=512) 
        self.bn3_0 = nn.BatchNorm1d(512)
        self.d3_PReLU = nn.PReLU(512)

        self.single3_1 = nn.Linear(in_features = 512, out_features = 256)
        self.bn3_1 = nn.BatchNorm1d(256)
        self.s31_PReLU = nn.PReLU(256)

        self.single3_2 = nn.Linear(in_features = 256, out_features = 128)
        self.bn3_2 = nn.BatchNorm1d(128)
        self.s32_PReLU = nn.PReLU(128)

        self.single3_3 = nn.Linear(in_features = 128, out_features=64)
        self.bn3_3 = nn.BatchNorm1d(64)
        self.s33_PReLU = nn.PReLU(64)

        # block 4
        self.double4 = nn.Linear(in_features = 512 + 64, out_features=256)
        self.bn4_0 = nn.BatchNorm1d(256)
        self.d4_PReLU = nn.PReLU(256)

        self.single4_1 = nn.Linear(in_features = 256, out_features = 128)
        self.bn4_1 = nn.BatchNorm1d(128)
        self.s41_PReLU = nn.PReLU(128)

        self.single4_2 = nn.Linear(in_features = 128, out_features = 64)
        self.bn4_2 = nn.BatchNorm1d(64)
        self.s42_PReLU = nn.PReLU(64)

        self.single4_3 = nn.Linear(in_features = 64, out_features=32)
        self.bn4_3 = nn.BatchNorm1d(32)
        self.s43_PReLU = nn.PReLU(32)

        # block 5
        self.double5 = nn.Linear(in_features = 256 + 32, out_features= 128)
        self.bn5_0 = nn.BatchNorm1d(128)
        self.d5_PReLU = nn.PReLU(128)

        self.single5_1 = nn.Linear(in_features = 128, out_features = 64)
        self.bn5_1 = nn.BatchNorm1d(64)
        self.s51_PReLU = nn.PReLU(64)

        self.single5_2 = nn.Linear(in_features = 64, out_features = 32)
        self.bn5_2 = nn.BatchNorm1d(32)
        self.s52_PReLU = nn.PReLU(32)

        self.single5_3 = nn.Linear(in_features = 32, out_features=16)
        self.bn5_3 = nn.BatchNorm1d(16)
        self.s53_PReLU = nn.PReLU(16)

        # block 6
        self.double6 = nn.Linear(in_features = 128 + 16, out_features=64)
        self.bn6_0 = nn.BatchNorm1d(64)
        self.d6_PReLU = nn.PReLU(64)

        self.single6_1 = nn.Linear(in_features = 64, out_features = 32)
        self.bn6_1 = nn.BatchNorm1d(32)
        self.s61_PReLU = nn.PReLU(32)

        self.single6_2 = nn.Linear(in_features = 32, out_features = 16)
        self.bn6_2 = nn.BatchNorm1d(16)
        self.s62_PReLU = nn.PReLU(16)

        self.single6_3 = nn.Linear(in_features = 16, out_features=8)
        self.bn6_3 = nn.BatchNorm1d(8)
        self.s63_PReLU = nn.PReLU(8)

        # block 7
        self.double7 = nn.Linear(in_features = 64 + 8, out_features=32)
        self.d7_PReLU = nn.PReLU(32)
        self.single7_1 = nn.Linear(in_features = 32, out_features=16)
        self.s71_PReLU = nn.PReLU(16)
        self.single7_2 = nn.Linear(in_features = 16, out_features=8)
        self.s72_PReLU = nn.PReLU(8)
        self.single7_3 = nn.Linear(in_features = 8, out_features=4)
        self.s73_PReLU = nn.PReLU(4)

        # block 8
        self.double8 = nn.Linear(in_features = 4 + 32, out_features=16)
        self.d8_PReLU = nn.PReLU(16)
        self.single8_1 = nn.Linear(in_features = 16, out_features = 8)
        self.s81_PReLU = nn.PReLU(8)
        self.single8_2 = nn.Linear(in_features = 8, out_features = 4)
        self.s82_PReLU = nn.PReLU(4)
        self.single8_3 = nn.Linear(in_features = 4, out_features=2)
        self.s83_PReLU = nn.PReLU(2)

        # block 9
        self.double9 = nn.Linear(in_features = 2 + 16, out_features=8)
        self.d9_PReLU = nn.PReLU(8)
        self.single9_1 = nn.Linear(in_features = 8, out_features = 4)
        self.s91_PReLU = nn.PReLU(4)
        self.single9_2 = nn.Linear(in_features = 4, out_features=2)
        self.s92_PReLU = nn.PReLU(2)
        self.single9_3 = nn.Linear(in_features = 2, out_features=1)


    def forward(self, input): 
        out1_0 = self.d1_PReLU(self.double1(input))
        out1_1 = self.s11_PReLU(self.single1_1(self.dropout(out1_0)))
        out1_2 = self.s12_PReLU(self.single1_2(self.dropout(out1_1)))
        out1_3 = self.s13_PReLU(self.single1_3(self.dropout(out1_2)))
        out1 = torch.cat((out1_0, out1_3), 1)

        out2_0 = self.d2_PReLU(self.double2(self.dropout(out1)))
        out2_1 = self.s21_PReLU(self.single2_1(self.dropout(out2_0)))
        out2_2 = self.s22_PReLU(self.single2_2(self.dropout(out2_1)))
        out2_3 = self.s23_PReLU(self.single2_3(self.dropout(out2_2)))
        out2 = torch.cat((out2_0, out2_3), 1)

        out3_0 = self.d3_PReLU(self.double3(self.dropout(out2)))
        out3_1 = self.s31_PReLU(self.single3_1(self.dropout(out3_0)))
        out3_2 = self.s32_PReLU(self.single3_2(self.dropout(out3_1)))
        out3_3 = self.s33_PReLU(self.single3_3(self.dropout(out3_2)))
        out3 = torch.cat((out3_0, out3_3), 1)

        out4_0 = self.d4_PReLU(self.double4(self.dropout(out3)))
        out4_1 = self.s41_PReLU(self.single4_1(self.dropout(out4_0)))
        out4_2 = self.s42_PReLU(self.single4_2(self.dropout(out4_1)))
        out4_3 = self.s43_PReLU(self.single4_3(self.dropout(out4_2)))
        out4 = torch.cat((out4_0, out4_3), 1)

        out5_0 = self.d5_PReLU(self.double5(self.dropout(out4)))
        out5_1 = self.s51_PReLU(self.single5_1(self.dropout(out5_0)))
        out5_2 = self.s52_PReLU(self.single5_2(self.dropout(out5_1)))
        out5_3 = self.s53_PReLU(self.single5_3(out5_2))
        out5 = torch.cat((out5_0, out5_3), 1)

        out6_0 = self.d6_PReLU(self.double6(self.dropout(out5)))
        out6_1 = self.s61_PReLU(self.single6_1(self.dropout(out6_0)))
        out6_2 = self.s62_PReLU(self.single6_2(out6_1))
        out6_3 = self.s63_PReLU(self.single6_3(out6_2))
        out6 = torch.cat((out6_0, out6_3), 1)

        out7_0 = self.d7_PReLU(self.double7(self.dropout(out6)))
        out7_1 = self.s71_PReLU(self.single7_1(out7_0))
        out7_2 = self.s72_PReLU(self.single7_2(out7_1))
        out7_3 = self.s73_PReLU(self.single7_3(out7_2))
        out7 = torch.cat((out7_0, out7_3), 1)

        out8_0 = self.d8_PReLU(self.double8(out7))
        out8_1 = self.s81_PReLU(self.single8_1(out8_0))
        out8_2 = self.s82_PReLU(self.single8_2(out8_1))
        out8_3 = self.s83_PReLU(self.single8_3(out8_2))
        out8 = torch.cat((out8_0, out8_3), 1)

        out9_0 = self.d9_PReLU(self.double9(out8))
        out9_1 = self.s91_PReLU(self.single9_1(out9_0))
        out9_2 = self.s92_PReLU(self.single9_2(out9_1))
        out9_3 = self.single9_3(out9_2)

        return out9_3

