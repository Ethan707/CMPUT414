'''
Author: Ethan Chen
Date: 2022-03-31 01:44:25
LastEditTime: 2022-03-31 01:44:26
LastEditors: Ethan Chen
Description: some code from the source, but do not use in the implementation
FilePath: \CMPUT414\abandon.py
'''
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class encoder_with_convs_and_symmetry(nn.Module):
    def __init__(self, input_size, output_size=256):
        super(encoder_with_convs_and_symmetry, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False, stride=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1, bias=False, stride=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, bias=False, stride=1)
        self.conv5 = nn.Conv1d(256, output_size, kernel_size=1, bias=False, stride=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = torch.max(x, 1)
        return x


class decoder_with_fc_only(nn.Module):
    def __init__(self, input_size):
        super(decoder_with_fc_only, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2048*3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class AE(nn.Module):
    def __init__(self, input_size, output_size=256):
        super(AE, self).__init__()
        self.encoder = encoder_with_convs_and_symmetry(input_size, output_size)
        self.decoder = decoder_with_fc_only(output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
