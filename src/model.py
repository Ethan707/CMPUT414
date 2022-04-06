'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-04 11:09:57
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


# Point cloud Net
# from learning3d
# https://github.com/vinits5/learning3d


class PCN(torch.nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc", num_coarse=1024, grid_size=4, detailed_output=True):
        # emb_dims:			Embedding Dimensions for PCN.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PCN, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.num_coarse = num_coarse
        self.detailed_output = detailed_output
        self.grid_size = grid_size
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.pooling = nn.MaxPool1d

        self.encoder()
        self.decoder_layers = self.decoder()
        if detailed_output:
            self.folding_layers = self.folding()

    def encoder_1(self):
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.relu = torch.nn.ReLU()

        # self.bn1 = torch.nn.BatchNorm1d(128)
        # self.bn2 = torch.nn.BatchNorm1d(256)

        layers = [self.conv1, self.relu, self.conv2]
        return layers

    def encoder_2(self):
        self.conv3 = torch.nn.Conv1d(2*256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, self.emb_dims, 1)

        # self.bn3 = torch.nn.BatchNorm1d(512)
        # self.bn4 = torch.nn.BatchNorm1d(self.emb_dims)
        self.relu = torch.nn.ReLU()

        layers = [self.conv3, self.relu,
                  self.conv4]
        return layers

    def encoder(self):
        self.encoder_layers1 = self.encoder_1()
        self.encoder_layers2 = self.encoder_2()

    def decoder(self):
        self.linear1 = torch.nn.Linear(self.emb_dims, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, self.num_coarse*3)

        # self.bn1 = torch.nn.BatchNorm1d(1024)
        # self.bn2 = torch.nn.BatchNorm1d(1024)
        # self.bn3 = torch.nn.BatchNorm1d(self.num_coarse*3)
        self.relu = torch.nn.ReLU()

        layers = [self.linear1, self.relu,
                  self.linear2, self.relu,
                  self.linear3]
        return layers

    def folding(self):
        self.conv5 = torch.nn.Conv1d(1029, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 512, 1)
        self.conv7 = torch.nn.Conv1d(512, 3, 1)

        # self.bn5 = torch.nn.BatchNorm1d(512)
        # self.bn6 = torch.nn.BatchNorm1d(512)
        self.relu = torch.nn.ReLU()

        layers = [self.conv5, self.relu,
                  self.conv6, self.relu,
                  self.conv7]
        return layers

    def fine_decoder(self):
        # Fine Output
        linspace = torch.linspace(-0.05, 0.05, steps=self.grid_size)
        grid = torch.meshgrid(linspace, linspace)
        grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2))								# 16x2
        grid = torch.unsqueeze(grid, dim=0)													# 1x16x2
        grid_feature = grid.repeat([self.coarse_output.shape[0], self.num_coarse, 1])		# Bx16384x2

        point_feature = torch.unsqueeze(self.coarse_output, dim=2)							# Bx1024x1x3
        point_feature = point_feature.repeat([1, 1, self.grid_size ** 2, 1])				# Bx1024x16x3
        point_feature = torch.reshape(point_feature, (-1, self.num_fine, 3))				# Bx16384x3

        global_feature = torch.unsqueeze(self.global_feature_v, dim=1)						# Bx1x1024
        global_feature = global_feature.repeat([1, self.num_fine, 1])						# Bx16384x1024

        feature = torch.cat([grid_feature, point_feature, global_feature], dim=2)			# Bx16384x1029

        center = torch.unsqueeze(self.coarse_output, dim=2)									# Bx1024x1x3
        center = center.repeat([1, 1, self.grid_size ** 2, 1])								# Bx1024x16x3
        center = torch.reshape(center, [-1, self.num_fine, 3])								# Bx16384x3

        output = feature.permute(0, 2, 1)
        for idx, layer in enumerate(self.folding_layers):
            output = layer(output)
        fine_output = output.permute(0, 2, 1) + center
        return fine_output

    def encode(self, input_data):
        output = input_data
        for idx, layer in enumerate(self.encoder_layers1):
            output = layer(output)

        global_feature_g = self.pooling(output)

        global_feature_g = global_feature_g.unsqueeze(2)
        global_feature_g = global_feature_g.repeat(1, 1, self.num_points)
        output = torch.cat([output, global_feature_g], dim=1)

        for idx, layer in enumerate(self.encoder_layers2):
            output = layer(output)

        self.global_feature_v = self.pooling(output)

    def decode(self):
        output = self.global_feature_v
        for idx, layer in enumerate(self.decoder_layers):
            output = layer(output)
        self.coarse_output = output.view(self.global_feature_v.shape[0], self.num_coarse, 3)

    def forward(self, input_data):
        # input_data: 		Point Cloud having shape input_shape.
        # output:			PointNet features (Batch x emb_dims)
        if self.input_shape == "bnc":
            self.num_points = input_data.shape[1]
            input_data = input_data.permute(0, 2, 1)
        else:
            self.num_points = input_data.shape[2]
        if input_data.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        self.encode(input_data)
        self.decode()

        result = {'coarse_output': self.coarse_output}

        if self.detailed_output:
            fine_output = self.fine_decoder()
            result['fine_output'] = fine_output

        return result

# classifier network


class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        # block 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # block 2
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # block 3
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # block 4
        self.conv8 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # block 5
        self.conv11 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.max_pool_5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # FC layers
        self.fc1 = nn.Linear(512*64, 4096)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, 30)

    def forward(self, input_data):
        # block 1
        print('0', input_data.shape)
        x = input_data.reshape(-1, 3, 2048)
        print('1', x.shape)
        x = F.relu(self.conv1(x))
        print('2', x.shape)
        x = F.relu(self.conv2(x))
        print('3', x.shape)
        x = self.max_pool_1(x)
        print('4', x.shape)

        # block 2
        x = F.relu(self.conv3(x))
        print('5', x.shape)
        x = F.relu(self.conv4(x))
        print('6', x.shape)
        x = self.max_pool_2(x)
        print('7', x.shape)

        # block 3
        x = F.relu(self.conv5(x))
        print('8', x.shape)
        x = F.relu(self.conv6(x))
        print('9', x.shape)
        x = F.relu(self.conv7(x))
        print('10', x.shape)
        x = self.max_pool_3(x)
        print('11', x.shape)

        # block 4
        x = F.relu(self.conv8(x))
        print('12', x.shape)
        x = F.relu(self.conv9(x))
        print('13', x.shape)
        x = F.relu(self.conv10(x))
        print('14', x.shape)
        x = self.max_pool_4(x)
        print('15', x.shape)

        # block 5
        x = F.relu(self.conv11(x))
        print('16', x.shape)
        x = F.relu(self.conv12(x))
        print('17', x.shape)
        x = F.relu(self.conv13(x))
        print('18', x.shape)
        x = self.max_pool_5(x)
        print('19', x.shape)

        # FC layers
        x = x.view(x.size(0), -1)
        print('20', x.shape)
        x = F.relu(self.fc1(x))
        print('21', x.shape)
        x = self.dropout_1(x)
        print('22', x.shape)
        x = F.relu(self.fc2(x))
        print('23', x.shape)
        x = self.dropout_2(x)
        print('24', x.shape)
        x = self.fc3(x)
        print('25', x.shape)
        return x
