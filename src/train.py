'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-03-31 18:43:43
LastEditors: Ethan Chen
Description: 
FilePath: /CMPUT414/src/train.py
'''
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import *
from model import *
from pytorch3d.loss import chamfer_distance
# def setup():

#     pass


# def train(args, io):
#     train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points),
#                               num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)


# def test(args, io):
#     test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
#                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)


# def parse_args():
#     parser = argparse.ArgumentParser(description="Arg parser")
#     parser.add_argument()

def modeltest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder()
    model.to(device)
    x = torch.randn(2048, 3).to(device)
    print(model(x).shape)


if __name__ == "__main__":
    #     args = parse_args()
    #     setup()
    modeltest()

    # num_epoch = 1
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # modelnet40 = ModelNet40(partition='train', num_points=1024)
    # train_loader = DataLoader(modelnet40,
    #                           num_workers=8, batch_size=50, shuffle=True, drop_last=True)
    # model = AutoEncoder()
    # model.to(device)

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
    #     print(train_loss)
    #   print(i)
    #   print(len(data))
