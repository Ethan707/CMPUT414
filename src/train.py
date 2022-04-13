'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-13 00:55:01
LastEditors: Ethan Chen
Description:
FilePath: /CMPUT414/src/train.py
'''
from unittest import result
from loss import *
from torchsummary import summary
from model import *
from data import *
import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.loss import cd_loss_L1, cd_loss_L2
from metrics.metric import l1_cd, l2_cd, f_score
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# TODO:
# 2. l-GAN


TRAIN_DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/train_files.txt'
TEST_DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/test_files.txt'
LOSS_MODEL_PATH_2048 = '/home/ethan/Code/Project/CMPUT414/model/loss_network_2048.pth'
LOSS_MODEL_PATH_128 = '/home/ethan/Code/Project/CMPUT414/model/loss_network_128.pth'
COMPLETION_MODEL_PATH = '/home/ethan/Code/Project/CMPUT414/model/completion_network_gridsize_4_CD_PN_loss.pth'
EVALUATE_RESULT_PATH = '/home/ethan/Code/Project/CMPUT414/results/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_result(name, path, shape, result_original, result_translated, result_rotated):
    with open(os.path.join(path, 'result.csv'), 'a') as f:
        csv_wite = csv.writer(f)
        csv_wite.writerow([name, SHAPE_NAME[shape], result_original[0], result_original[1], result_original[2],
                          result_translated[0], result_translated[1], result_translated[2],
                          result_rotated[0], result_rotated[1], result_rotated[2]])


def init_result(path):
    with open(os.path.join(path, 'result.csv'), 'w') as f:
        csv_wite = csv.writer(f)
        csv_wite.writerow(['model', 'shape', 'original_l1_cd', 'original_l2_cd', 'original_f1', 'translated_l1_cd',
                          'translated_l2_cd', 'translated_f1', 'rotated_l1_cd', 'rotated_l2_cd', 'rotated_f1'])


def train_loss_network(model, path, num_epoch=100, num_points=2048):
    modelnet40_train = ModelNet40(num_points=num_points, data_path=TRAIN_DATA_PATH)
    modelnet40_test = ModelNet40(num_points=num_points, data_path=TEST_DATA_PATH)
    train_loader = DataLoader(modelnet40_train, num_workers=8, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=32, shuffle=True, drop_last=True)

    model.to(DEVICE)
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
            ground_truth = original[0].to(DEVICE)
            label = label.to(DEVICE).squeeze()
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
                ground_truth = original[0].to(DEVICE)
                label = label.to(DEVICE).squeeze()
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


def train_completion_network(model, loss1, loss2, path, use_augmentation=False, num_epoch=400):
    modelnet40_train = ModelNet40(num_points=2048, data_path=TRAIN_DATA_PATH)
    modelnet40_test = ModelNet40(num_points=2048, data_path=TEST_DATA_PATH)

    train_loader = DataLoader(modelnet40_train, num_workers=8, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=32, shuffle=True, drop_last=True)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    best_l1_cd = 1e8
    best_epoch = 0
    train_step = 0

    print("Total epoch: {}".format(num_epoch))
    for epoch in range(num_epoch):
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif epoch < 50000:
            alpha = 0.5
        else:
            alpha = 1.0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            original, translated, rotated, _ = data

            # original data
            cutout = original[2].to(DEVICE)
            optimizer.zero_grad()
            coarse_pred, dense_pred = model(cutout)
            coarse_loss = loss1(coarse_pred, original[1].to(DEVICE))
            dense_loss = loss2(dense_pred, original[0].to(DEVICE))
            cd_1 = cd_loss_L1(coarse_pred, original[1].to(DEVICE))
            cd_2 = cd_loss_L1(dense_pred, original[0].to(DEVICE))
            loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
            loss.backward()
            optimizer.step()

            if use_augmentation:
                cutout = translated[2].to(DEVICE)
                optimizer.zero_grad()
                coarse_pred, dense_pred = model(cutout)
                coarse_loss = loss1(coarse_pred, translated[1].to(DEVICE))
                dense_loss = loss2(dense_pred, translated[0].to(DEVICE))
                cd_1 = cd_loss_L1(coarse_pred, translated[1].to(DEVICE))
                cd_2 = cd_loss_L1(dense_pred, translated[0].to(DEVICE))
                loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
                loss.backward()
                optimizer.step()

                cutout = rotated[2].to(DEVICE)
                optimizer.zero_grad()
                coarse_pred, dense_pred = model(cutout)
                coarse_loss = loss1(coarse_pred, rotated[1].to(DEVICE))
                dense_loss = loss2(dense_pred, rotated[0].to(DEVICE))
                cd_1 = cd_loss_L1(coarse_pred, rotated[1].to(DEVICE))
                cd_2 = cd_loss_L1(dense_pred, rotated[0].to(DEVICE))
                loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
                loss.backward()
                optimizer.step()

            train_step += 1
        scheduler.step()
        print("epoch: {}, loss1: {}, loss2: {}, loss: {}".format(
            epoch, coarse_loss.item(), dense_loss.item(), loss.item()))

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                original, translated, rotated, _ = data
                cutout = original[2].to(DEVICE)
                coarse_pred, dense_pred = model(cutout)
                test_loss += l1_cd(dense_pred, original[0].to(DEVICE))
            print("L1 CD: {}".format(test_loss / len(test_loader.dataset)*1e3))
        if test_loss / len(test_loader.dataset) < best_l1_cd:
            best_l1_cd = test_loss / len(test_loader.dataset)
            best_epoch = epoch
            torch.save(model.state_dict(), path)
    print("Best L1 CD: {}, at epoch: {}".format(best_l1_cd, best_epoch))


def evaluate_single(model, pc):
    _, pred = model(pc[2].to(DEVICE))
    f1, l1cd, l2cd = 0, 0, 0

    l1cd += l1_cd(pred, pc[0].to(DEVICE)).item()
    l2cd += l2_cd(pred, pc[0].to(DEVICE)).item()

    # print(len(pc[0]))
    for i in range(len(pc[0])):
        single_pc = pc[0][i].detach().cpu().numpy()
        single_pred = pred[i].detach().cpu().numpy()
        # print(single_pc.shape)
        f1 += f_score(single_pred, single_pc)
    return (l1cd, l2cd, f1)


def evaluate_single_shape(shape, model):
    modelnet40_test = ModelNet40(num_points=2048, data_path=TEST_DATA_PATH, shape=shape)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=32, shuffle=True)

    # make dir
    # shape_dir = os.path.join(path, shape)
    # image_dir = os.path.join(shape_dir, 'images')
    # output_dir = os.path.join(shape_dir, 'output')
    # os.makedirs(shape_dir)
    # os.makedirs(image_dir)
    # os.makedirs(output_dir)

    # evaluate model

    original_l1_cd, original_l2_cd, original_f1 = 0, 0, 0
    translated_l1_cd, translated_l2_cd, translated_f1 = 0, 0, 0
    rotated_l1_cd, rotated_l2_cd, rotated_f1 = 0, 0, 0
    length = len(test_loader.dataset)

    with torch.no_grad():
        for original, translated, rotated, _ in test_loader:
            # original
            result = evaluate_single(model, original)
            original_l1_cd += result[0]
            original_l2_cd += result[1]
            original_f1 += result[2]

            # translated
            result = evaluate_single(model, translated)
            translated_l1_cd += result[0]
            translated_l2_cd += result[1]
            translated_f1 += result[2]

            # rotated
            result = evaluate_single(model, rotated)
            rotated_l1_cd += result[0]
            rotated_l2_cd += result[1]
            rotated_f1 += result[2]

    ave_original_l1_cd = original_l1_cd / length
    ave_original_l2_cd = original_l2_cd / length
    ave_original_f1 = original_f1 / length

    ave_translated_l1_cd = translated_l1_cd / length
    ave_translated_l2_cd = translated_l2_cd / length
    ave_translated_f1 = translated_f1 / length

    ave_rotated_l1_cd = rotated_l1_cd / length
    ave_rotated_l2_cd = rotated_l2_cd / length
    ave_rotated_f1 = rotated_f1 / length

    result_original = (ave_original_l1_cd, ave_original_l2_cd, ave_original_f1)
    result_translated = (ave_translated_l1_cd, ave_translated_l2_cd, ave_translated_f1)
    result_rotated = (ave_rotated_l1_cd, ave_rotated_l2_cd, ave_rotated_f1)

    return result_original, result_translated, result_rotated


def evaluate_model(completion_model, name):

    path = os.path.join(EVALUATE_RESULT_PATH, datetime.now().strftime("%m-%d-%H-%M-%S"))
    os.makedirs(path)
    print("Evaluate model: {}".format(path))
    init_result(path)

    completion_model.eval()
    for shape in range(40):
        print("Shape: {}".format(SHAPE_NAME[shape]))
        result_original, result_translated, result_rotated = evaluate_single_shape(shape, completion_model)
        write_result(name, path, shape, result_original, result_translated, result_rotated)


if __name__ == "__main__":
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # model
    loss_2048 = PointNetCls(k=40)
    loss_128 = PointNetCls(k=40)
    completion_network = PCN(num_dense=2048, grid_size=4)
    continue_training = True

    # arguments
    num_epoch = 400
    # load & train model
    if not os.path.exists(LOSS_MODEL_PATH_2048):
        print("Training loss network 2048...")
        train_loss_network(loss_2048, path=LOSS_MODEL_PATH_2048, num_epoch=num_epoch, num_points=2048)
    loss_2048.load_state_dict(torch.load(LOSS_MODEL_PATH_2048))
    loss_2048.to(DEVICE)
    print("Loaded loss network 2048.")

    if not os.path.exists(LOSS_MODEL_PATH_128):
        print("Training loss network 128...")
        train_loss_network(loss_128, path=LOSS_MODEL_PATH_128, num_epoch=num_epoch, num_points=128)
    loss_128.load_state_dict(torch.load(LOSS_MODEL_PATH_128))
    loss_128.to(DEVICE)
    print("Loaded loss network 128.")

    dense_loss = PointNetLoss(loss_2048, DEVICE, 0.1)
    coarse_loss = PointNetLoss(loss_128, DEVICE, 0.1)

    # if not os.path.exists(COMPLETION_MODEL_PATH) or continue_training:
    #     print("Training completion network...")
    #     if continue_training:
    #         print("Continue training completion network...")
    #         completion_network.load_state_dict(torch.load(COMPLETION_MODEL_PATH))
    #         completion_network.to(DEVICE)
    #     train_completion_network(completion_network, path=COMPLETION_MODEL_PATH,
    #                              loss1=coarse_loss, loss2=dense_loss, num_epoch=num_epoch)
    #     print("Training finished.")
    # completion_network.load_state_dict(torch.load(COMPLETION_MODEL_PATH))
    # completion_network.to(DEVICE)
    # print("Loaded completion network.")

    # evaluate model and save results
    # evaluate_model(completion_network, "PCN_gridsize_1_CD_PN")
    # print("Evaluation finished.")

    EXPERIMENT = [
        "PCN_gridsize_4_CD_AUGMENT",
        "PCN_gridsize_4_CD_PN_AUGMENT",
    ]

    for experiment in EXPERIMENT:
        completion_network = PCN(num_dense=2048, grid_size=4)
        COMPLETION_MODEL_PATH = '/home/ethan/Code/Project/CMPUT414/model/{}.pth'.format(experiment)
        train_completion_network(completion_network, path=COMPLETION_MODEL_PATH,
                                 loss1=coarse_loss, loss2=dense_loss, num_epoch=num_epoch, use_augmentation=True)
        evaluate_model(completion_network, experiment)
        print(experiment+"Evaluation finished.")
