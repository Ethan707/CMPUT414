'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-14 12:05:41
LastEditors: Ethan Chen
Description:
FilePath: /CMPUT414/src/train.py
'''
import sys
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
# 1. different severity
# 2. implement different model (GAN)
# 3. try different feature map
# 4. implement different dataset (completion3d or shapeNet)


TRAIN_DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/train_files.txt'
TEST_DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/test_files.txt'
LOSS_MODEL_PATH_2048 = '/home/ethan/Code/Project/CMPUT414/model/loss_network_2048.pth'
LOSS_MODEL_PATH_128 = '/home/ethan/Code/Project/CMPUT414/model/loss_network_128.pth'
COMPLETION_MODEL_PATH = '/home/ethan/Code/Project/CMPUT414/model/completion_network_gridsize_4_CD_PN_loss.pth'
EVALUATE_RESULT_PATH = '/home/ethan/Code/Project/CMPUT414/results/'
EXPERIMENT = ''


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


def train_loss_network(model, args):
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


def train_completion_network(model, loss1, loss2, args):
    if args.dataset == 'modelnet40':
        train_set = ModelNet40(num_points=2048, data_path=TRAIN_DATA_PATH)
        test_set = ModelNet40(num_points=2048, data_path=TEST_DATA_PATH)
    elif args.dataset == 'shapenet':
        train_set = ShapeNet()
        test_set = ShapeNet()
        validate_set = ShapeNet()
    # TODO: add more dataset
    train_loader = DataLoader(train_set, num_workers=8, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, num_workers=8, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    best_l1_cd = 1e8
    best_epoch = 0
    train_step = 0

    print("Total epoch: {}".format(args.epochs))
    for epoch in range(args.epochs):
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif epoch < 50000:
            alpha = 0.5
        else:
            alpha = 1.0
        model.train()

        for _, data in enumerate(tqdm(train_loader)):
            original, translated, rotated, _ = data

            # original data
            cutout = original[2].to(DEVICE)
            optimizer.zero_grad()
            coarse_pred, dense_pred = model(cutout)

            cd_1 = cd_loss_L1(coarse_pred, original[1].to(DEVICE))
            cd_2 = cd_loss_L1(dense_pred, original[0].to(DEVICE))
            if use_loss:
                coarse_loss = loss1(coarse_pred, original[1].to(DEVICE))
                dense_loss = loss2(dense_pred, original[0].to(DEVICE))
                loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
            else:
                loss = cd_2 + alpha * cd_1
            loss.backward()
            optimizer.step()

            if use_augmentation:
                cutout = translated[2].to(DEVICE)
                optimizer.zero_grad()
                coarse_pred, dense_pred = model(cutout)

                cd_1 = cd_loss_L1(coarse_pred, translated[1].to(DEVICE))
                cd_2 = cd_loss_L1(dense_pred, translated[0].to(DEVICE))
                if use_loss:
                    coarse_loss = loss1(coarse_pred, translated[1].to(DEVICE))
                    dense_loss = loss2(dense_pred, translated[0].to(DEVICE))
                    loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
                else:
                    loss = cd_2 + alpha * cd_1
                loss.backward()
                optimizer.step()

                cutout = rotated[2].to(DEVICE)
                optimizer.zero_grad()
                coarse_pred, dense_pred = model(cutout)

                cd_1 = cd_loss_L1(coarse_pred, rotated[1].to(DEVICE))
                cd_2 = cd_loss_L1(dense_pred, rotated[0].to(DEVICE))
                if use_loss:
                    coarse_loss = loss1(coarse_pred, rotated[1].to(DEVICE))
                    dense_loss = loss2(dense_pred, rotated[0].to(DEVICE))
                    loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
                else:
                    loss = cd_2 + alpha * cd_1

                loss.backward()
                optimizer.step()

            train_step += 1
        scheduler.step()
        # print("epoch: {}, loss1: {}, loss2: {}, loss: {}".format(
        #     epoch, coarse_loss.item(), dense_loss.item(), loss.item()))

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(test_loader)):
                original, translated, rotated, _ = data
                cutout = original[2].to(DEVICE)
                coarse_pred, dense_pred = model(cutout)
                test_loss += l1_cd(dense_pred, original[0].to(DEVICE))
            print("epoch {}, L1 CD: {}".format(epoch, test_loss / len(test_loader.dataset)*1e3))
        if test_loss / len(test_loader.dataset) < best_l1_cd:
            best_l1_cd = test_loss / len(test_loader.dataset)
            best_epoch = epoch
            torch.save(model.state_dict(), path)
    print("Best L1 CD: {}, at epoch: {}".format(best_l1_cd, best_epoch))


def evaluate_single(model, pc, args):
    _, pred = model(pc[2].to(args.device))
    f1, l1cd, l2cd = 0, 0, 0

    l1cd += l1_cd(pred, pc[0].to(args.device)).item()
    l2cd += l2_cd(pred, pc[0].to(args.device)).item()

    # print(len(pc[0]))
    for i in range(len(pc[0])):
        single_pc = pc[0][i].detach().cpu().numpy()
        single_pred = pred[i].detach().cpu().numpy()
        # print(single_pc.shape)
        f1 += f_score(single_pred, single_pc)
    return (l1cd, l2cd, f1)


def evaluate_single_shape(shape, model, args):
    modelnet40_test = ModelNet40(num_points=2048, data_path=TEST_DATA_PATH, shape=shape)
    test_loader = DataLoader(modelnet40_test, num_workers=8, batch_size=32, shuffle=True)

    # evaluate model

    original_l1_cd, original_l2_cd, original_f1 = 0, 0, 0
    translated_l1_cd, translated_l2_cd, translated_f1 = 0, 0, 0
    rotated_l1_cd, rotated_l2_cd, rotated_f1 = 0, 0, 0
    length = len(test_loader.dataset)

    with torch.no_grad():
        for original, translated, rotated, _ in test_loader:
            # original
            result = evaluate_single(model, original, args)
            original_l1_cd += result[0]
            original_l2_cd += result[1]
            original_f1 += result[2]

            # translated
            result = evaluate_single(model, translated, args)
            translated_l1_cd += result[0]
            translated_l2_cd += result[1]
            translated_f1 += result[2]

            # rotated
            result = evaluate_single(model, rotated, args)
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


def evaluate_model(completion_model, args):

    path = os.path.join(EVALUATE_RESULT_PATH, args.experiment_name)
    if os.path.exists(path):
        path = os.path.join(path, datetime.now().strftime('%m-%d-%H-%M-%S'))
    os.makedirs(path)
    print("Evaluate model: {}".format(path))
    init_result(path)

    completion_model.eval()
    if args.dataset == 'modelnet40':
        for shape in range(40):
            print("Shape: {}".format(SHAPE_NAME[shape]))
            result_original, result_translated, result_rotated = evaluate_single_shape(shape, completion_model, args)
            write_result(args.experiment_name, path, shape, result_original, result_translated, result_rotated)
    elif args.dataset == 'shapenet':
        for shape in range(len(SHAPE_NAME)):
            print("Shape: {}".format(SHAPE_NAME[shape]))
            result_original, result_translated, result_rotated = evaluate_single_shape(shape, completion_model, args)
            write_result(args.experiment_name, path, shape, result_original, result_translated, result_rotated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=400, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_model', action='store_true', help='For Saving the current Model')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--use_augmentation', action='store_true', help='For data augmentation')
    parser.add_argument('--loss_net', type=str, default='PointNet', help='loss network')
    parser.add_argument('--experiment', type=str, default='', help='experiment name')
    parser.add_argument('--action', type=str, default='train', help='action name[train/evaluate]')

    args = parser.parse_args()

    # set up seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # set up experiment name
    if args.experiment == '':
        args.experiment = 'PCN_'+args.loss_net+'_'+args.dataset+'_'+str(args.epochs)+'_'+str(args.seed)
    if args.use_augmentation:
        args.experiment += '_augmentation'
    print("Experiment: {}".format(args.experiment))

    assert args.action in ['train', 'evaluate']

    # TODO:
    if args.action == 'evaluate':
        # load model
        if not os.path.exists():
            pass
        # evaluate_model(, args.experiment)
        sys.exit(0)

    # set up or train loss network (PointNet or DGCNN)
    if args.loss_net == 'PointNet':
        if args.dataset == 'modelnet40':
            loss_net_dense = PointNetCls(k=40)
            loss_net_coarse = PointNetCls(k=40)

            # if not exist pretrained model, train from scratch
            if not os.path.exists(os.path.join('')):
                train_loss_network()
            else:
                loss_net_dense.load_state_dict(torch.load(os.path.join('')))
                loss_net_coarse.load_state_dict(torch.load(os.path.join('')))

            dense_loss = PointNetLoss(loss_net_dense, args.device, 0.1)
            coarse_loss = PointNetLoss(loss_net_coarse, args.device, 0.1)

        elif args.dataset == 'shapenet':
            loss_net_dense = ...
            loss_net_coarse = ...

            # if not exist pretrained model, train from scratch
            if not os.path.exists(os.path.join('')):
                train_loss_network()
            else:
                loss_net_dense.load_state_dict(torch.load(os.path.join('')))
                loss_net_coarse.load_state_dict(torch.load(os.path.join('')))
            dense_loss = PointNetLoss(loss_net_dense, args.device, 0.1)
            coarse_loss = PointNetLoss(loss_net_coarse, args.device, 0.1)
        else:
            assert False, "Dataset not supported"

    elif args.loss_net == 'DGCNN':
        if args.dataset == 'modelnet40':
            loss_net_dense = ...
            loss_net_coarse = ...

            # if not exist pretrained model, train from scratch
            if not os.path.exists(os.path.join('')):
                pass
            else:
                loss_net_dense.load_state_dict(torch.load(os.path.join('')))
                loss_net_coarse.load_state_dict(torch.load(os.path.join('')))
        elif args.dataset == 'shapenet':
            loss_net_dense = ...
            loss_net_coarse = ...

            # if not exist pretrained model, train from scratch
            if not os.path.exists(os.path.join('')):
                pass
            else:
                loss_net_dense.load_state_dict(torch.load(os.path.join('')))
                loss_net_coarse.load_state_dict(torch.load(os.path.join('')))
        else:
            assert False, "Dataset not supported"

    elif args.loss_net == 'None':
        dense_loss = None
        coarse_loss = None
    else:
        assert False, "Loss network not supported"

    print("Loss network: {}".format(args.loss_net))

    # set up completion model
    if args.dataset == 'modelnet40':
        completion_network = PCN(num_dense=2048).to(args.device)
    else:
        completion_network = PCN().to(args.device)

    train_completion_network(completion_network, coarse_loss, dense_loss, args)
    print("Training completed")

    evaluate_model(completion_network, args)
    print(args.experiment, "Evaluation completed")
