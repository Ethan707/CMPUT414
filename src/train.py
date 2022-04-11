'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-11 02:09:19
LastEditors: Ethan Chen
Description:
FilePath: /CMPUT414/src/train.py
'''
from loss import *
from torchsummary import summary
from pytorch3d.loss import chamfer_distance
from model import *
from data import *
import argparse
import os
import random
from statistics import mode
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO:
# 1. chamer_distance + classification loss
# 2. add the laplace GAN
# 2. add a discriminator for the classifier
# 3. use different model to train the completion network
# 4. train with different class
# 5. train with different batch size
# 6. train with different learning rate
# 7. train with different epoch
TRAIN_DATA_PATH = '/home/ethan/Code/1/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/train_files.txt'
TEST_DATA_PATH = '/home/ethan/Code/1/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/test_files.txt'
LOSS_MODEL_PATH_2048 = '/home/ethan/Code/1/Project/CMPUT414/model/loss_network_2048.pth'
LOSS_MODEL_PATH_128 = '/home/ethan/Code/1/Project/CMPUT414/model/loss_network_128.pth'
COMPLETION_MODEL_PATH = '/home/ethan/Code/1/Project/CMPUT414/model/completion_network.pth'


def train_loss_network(model, path, num_epoch=100, num_points=2048, device='cuda'):
    modelnet40_train = ModelNet40(num_points=num_points, data_path=TRAIN_DATA_PATH)
    modelnet40_test = ModelNet40(num_points=num_points, data_path=TEST_DATA_PATH)
    train_loader = DataLoader(modelnet40_train, num_workers=8, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=32, shuffle=True, drop_last=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epoch):
        train_loss = 0
        train_correct = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            original, translated, rotated, label = data
            ground_truth = original[0].to(device)
            label = label.to(device).squeeze()
            optimizer.zero_grad()
            output, trans, trans_feat, feature_map = model(ground_truth)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == label).sum().item()
        scheduler.step()
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                original, translated, rotated, label = data
                ground_truth = original[0].to(device)
                label = label.to(device).squeeze()
                output, trans, trans_feat, feature_map = model(ground_truth)

                loss = criterion(output, label)
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                test_correct += (pred == label).sum().item()

        print("epoch: {}, train loss: {}, train accuracy: {}, test loss: {}, test accuracy: {}".format(
            epoch, train_loss/len(train_loader.dataset),
            train_correct / len(train_loader.dataset),
            test_loss / len(test_loader.dataset),
            test_correct / len(test_loader.dataset)))

        if test_correct / len(test_loader.dataset) > best_acc:
            best_acc = test_correct / len(test_loader.dataset)
            best_epoch = epoch
            torch.save(model.state_dict(), path)

    print("Best accuracy: {}, at epoch: {}".format(best_acc, best_epoch))


def train_completion_network(model, loss_model, path, num_epoch=100, num_points=2048, device='cuda'):
    modelnet40_train = ModelNet40(num_points=num_points, data_path=TRAIN_DATA_PATH)
    modelnet40_test = ModelNet40(num_points=num_points, data_path=TEST_DATA_PATH)
    train_loader = DataLoader(modelnet40_train, num_workers=8, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=32, shuffle=True, drop_last=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epoch):
        train_loss = 0
        train_correct = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            original, translated, rotated, label = data
            cutout = original[0].to(device)
            label = label.to(device).squeeze()
            optimizer.zero_grad()
            coarse_pred, dense_pred = model(cutout)
            print(coarse_pred.shape, dense_pred.shape)
            output, trans, trans_feat, feature_map = loss_model(dense_pred)
            print(feature_map.shape)


def modeltest(model):
    pass


if __name__ == "__main__":
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # model
    loss_network = PointNetCls(k=40)
    loss_network_128 = PointNetCls(k=40)
    completion_network = PCN(num_dense=2048)

    # arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 100
    # load & train model
    if not os.path.exists(LOSS_MODEL_PATH_2048):
        train_loss_network(loss_network, path=LOSS_MODEL_PATH_2048, num_epoch=num_epoch, num_points=2048, device=device)
    loss_network.load_state_dict(torch.load(LOSS_MODEL_PATH_2048))
    loss_network.to(device)

    train_loss_network(loss_network_128, path=LOSS_MODEL_PATH_128, num_epoch=num_epoch, num_points=128, device=device)

    # train_completion_network(completion_network, path=None, loss_model=loss_network, num_epoch=num_epoch, device=device)

    # modeltest(completion_network)

    # model_dict = torch.load('/home/ethan/Code/1/Project/CMPUT414/model/cls_model_epoch_260.pth')
    # loss_network.load_state_dict(model_dict)
    # loss_network.cuda()
    # loss_network.eval()
    # summary(loss_network, (3, 2048))
    # te.list_module_names(loss_network)
    # for name, module in loss_network.named_modules():
    #     print(name)
    # dummy_input = torch.randn(1, 3, 2048).cuda()
    # model = te.Extractor(loss_network, ["feat.stn.conv1"])
    # model_output, features = model(dummy_input)
    # print({name: feature.shape for name, feature in features.items()})
    # test_feature = loss_network.features
    # test_feature.eval()
    # loss_network = ...

    # if use_pretrained_completion:
    #     model_dict = torch.load('/home/ethan/Code/1/Project/CMPUT414/model/com_model_epoch_260.pth')
    #     completion_network.load_state_dict(model_dict)
    # else:
    #     train_completion_network(completion_network, loss_network)
