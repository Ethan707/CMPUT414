'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-11 01:47:39
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
        self.linear2 = nn.Linear(1024, 1024)

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
        x = F.relu(self.linear1(x))
        print(x.shape)
        x = F.relu(self.linear2(x))
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


# laplacian completion network


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1,
                                                                                      self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size,
                                                                                      1).expand(self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape

        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]                           # (B, 1024)

        # decoder
        # (B, num_coarse, 3), coarse point cloud
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)

        # (B, 3, num_fine), fine point cloud
        fine = self.final_conv(feat) + point_feat

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()
