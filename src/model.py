'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-01 07:45:02
LastEditors: Ethan Chen
Description: The model class for the latent 3d model
FilePath: /CMPUT414/src/model.py
'''
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# Learning Representations and Generative Models for 3D Point Clouds
# doi: 1707.02392

# shape completion network
# for each class specific model, train for a maximum of 100 epochs with Adam optimizer
# and a learning rate of 0.0005, batch size 50.

# Difference: the output size change from 4096 to 2048, and the input size change from 2048 to 1024


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False, stride=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False, stride=1)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024*3)
        self.linear3 = nn.Linear(1024*3, 2048*3)

    def forward(self, x):
        print(x.shape)
        x = x.reshape(-1, 3, 2048)
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = self.max_pool(x)
        print(x.shape)
        x = x.flatten(start_dim=1)
        print(x.shape)
        x = F.relu(self.linear1(x))
        print(x.shape)
        x = F.relu(self.linear2(x))
        print(x.shape)
        x = self.linear3(x)
        print(x.shape)
        return x

# raw GAN
# shape completion network r-GAN


class r_GAN_generator(nn.Module):
    def __init__(self):
        super(r_GAN_generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(3, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class r_GAN_discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(r_GAN_discriminator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(256),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x

# Point cloud Net


class PCN(nn.Module):
    def __init__(self, input_size, output_size=256):
        super(PCN, self).__init__()
        self.encode = self.encoder()
        self.decode = self.decoder()

    def encoder(self, x):
        pass

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
