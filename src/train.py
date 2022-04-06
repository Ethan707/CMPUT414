'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-04 11:34:46
LastEditors: Ethan Chen
Description: 
FilePath: /CMPUT414/src/train.py
'''
import argparse
import os
from statistics import mode
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import *
from model import *
from pytorch3d.loss import chamfer_distance
from torchsummary import summary

# TODO:
# 1. chamer_distance + classification loss
# 2. add the laplace GAN
# 2. add a discriminator for the classifier
# 3. use different model to train the completion network
# 4. train with different class
# 5. train with different batch size
# 6. train with different learning rate
# 7. train with different epoch


def modeltest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AutoEncoder()
    model = Classifier()
    model.to(device)
    x = torch.randn(2048, 3).to(device)
    model(x)
    summary(model, (2048, 3))


if __name__ == "__main__":
    #     args = parse_args()
    #     setup()
    modeltest()

    num_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelnet40 = ModelNet40(partition='train', num_points=1024)
    train_loader = DataLoader(modelnet40,
                              num_workers=8, batch_size=50, shuffle=True, drop_last=True)
    model = Classifier()
    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # train_loss = 0
    # for epoch in range(num_epoch):
    #     model.train()
    #     for i, data in enumerate(tqdm(train_loader)):
    #         ground_truth, pointcloud, label = data
    #         ground_truth = ground_truth.to(device)
    #         pointcloud = pointcloud.to(device)
    #         label = label.to(device)
    #         optimizer.zero_grad()
    #         output = model(pointcloud)
    #         loss = chamfer_distance(output, ground_truth)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #     print("epoch: {}, loss: {}".format(epoch, train_loss))
