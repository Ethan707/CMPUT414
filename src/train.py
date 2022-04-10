'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-09 17:38:03
LastEditors: Ethan Chen
Description: 
FilePath: /CMPUT414/src/train.py
'''
import argparse
import os
import random
from statistics import mode
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import *
from model import *
from pytorch3d.loss import chamfer_distance
from torchsummary import summary
from loss import *
# TODO:
# 1. chamer_distance + classification loss
# 2. add the laplace GAN
# 2. add a discriminator for the classifier
# 3. use different model to train the completion network
# 4. train with different class
# 5. train with different batch size
# 6. train with different learning rate
# 7. train with different epoch


def train_loss_network(model):
    num_epoch = 1001
    modelnet40_train = ModelNet40(partition='train', num_points=2048,
                                  data_path='/home/ethan/Code/1/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/train_files.txt')
    modelnet40_test = ModelNet40(partition='test', num_points=2048,
                                 data_path='/home/ethan/Code/1/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/test_files.txt')
    train_loader = DataLoader(modelnet40_train,
                              num_workers=8, batch_size=100, shuffle=True, drop_last=True)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=100, shuffle=True, drop_last=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        train_loss = 0
        correct = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            ground_truth, (cutout_pointcloud, translatedCloud, noiseCloud, rotatedCloud), label = data
            # ground_truth train
            ground_truth = ground_truth.to(device).permute(0, 2, 1)
            label = label.to(device).squeeze()
            optimizer.zero_grad()
            output, trans, trans_feat = model(ground_truth)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()

        if epoch % 10 == 0:
            print("Train epoch: {}, loss: {}, accuracy: {}".format(
                epoch, train_loss/len(train_loader.dataset), correct / len(train_loader.dataset)))

            model.eval()
            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for i, data in enumerate(tqdm(test_loader)):
                    ground_truth, label = data
                    ground_truth = ground_truth.to(device).permute(0, 2, 1)
                    label = label.to(device).squeeze()
                    output, trans, trans_feat = model(ground_truth)

                    loss = criterion(output, label)
                    test_loss += loss.item()
                    pred = output.argmax(dim=1)
                    test_correct += (pred == label).sum().item()
            print("Test epoch: {}, loss: {}, accuracy: {}".format(epoch, test_loss /
                  len(test_loader.dataset), test_correct / len(test_loader.dataset)))
            torch.save(model.state_dict(), '/home/ethan/Code/1/Project/CMPUT414/model/cls_model_epoch_{}.pth'.format(epoch))


if __name__ == "__main__":
    random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_network = PointNetCls(k=40)
    # loss_network = PointNet(1024, 40)

    train_loss_network(loss_network)
