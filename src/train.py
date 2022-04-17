'''
Author: Ethan Chen
Date: 2022-03-16 17:47:49
LastEditTime: 2022-04-16 23:46:54
LastEditors: Ethan Chen
Description:
FilePath: /CMPUT414/src/train.py
'''
import sys
from loss import *
from model import *
from data import ShapeNet, ModelNet40
import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.loss import cd_loss_L1, cd_loss_L2
from metrics.metric import l1_cd, l2_cd

# TODO:
# 1. different severity
# 2. implement different model (GAN)
# 3. try different feature map
# 4. implement different dataset (completion3d or shapeNet)

MODEL_PATH = '/home/ethan/Code/Project/CMPUT414/model'
DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data'


def train_loss_shapenet(model, num_points, args, epochs=100):
    train_path = os.path.join(DATA_PATH, '')
    test_path = os.path.join(DATA_PATH, '')
    train_set = ShapeNet()
    test_set = ShapeNet()
    train_loader = DataLoader(train_set, num_workers=8, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, num_workers=8, batch_size=8, shuffle=True)

    if args.loss_net == 'PointNet':
        pass
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_epoch = 0
    model.to(args.device)
    for epoch in range(epochs):
        pass


def train_loss_modelnet40(model, num_points, args, epochs=100):
    if args.loss_net == 'PointNet':
        batch_size = 32
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        batch_size = 8
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.001)

    train_path = os.path.join(DATA_PATH, 'modelnet40_ply_hdf5_2048/train_files.txt')
    test_path = os.path.join(DATA_PATH, 'modelnet40_ply_hdf5_2048/test_files.txt')
    train_set = ModelNet40(num_points=num_points, data_path=train_path, augmentation=args.augmentation)
    test_set = ModelNet40(num_points=num_points, data_path=test_path, augmentation=args.augmentation)
    train_loader = DataLoader(train_set, num_workers=8, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, num_workers=8, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_epoch = 0
    model.to(args.device)
    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0
        model.train()
        length_train = 0
        for _, data in enumerate(tqdm(train_loader)):
            if args.augmentation:
                original, translated, rotated, label = data

                gt = translated[0].to(args.device)
                label = label.to(args.device).squeeze()
                optimizer.zero_grad()
                if args.loss_net == 'PointNet':
                    output, _, _, _ = model(gt)
                else:
                    output, _ = model(gt)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += (pred == label).sum().item()

                gt = rotated[0].to(args.device)
                label = label.to(args.device).squeeze()
                optimizer.zero_grad()
                if args.loss_net == 'PointNet':
                    output, _, _, _ = model(gt)
                else:
                    output, _ = model(gt)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += (pred == label).sum().item()
                length_train += 2
            else:
                original, label = data

            gt = original[0].to(args.device)
            label = label.to(args.device).squeeze()
            optimizer.zero_grad()
            if args.loss_net == 'PointNet':
                output, _, _, _ = model(gt)
            else:
                output, _ = model(gt)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == label).sum().item()
            length_train += 1

        scheduler.step()
        model.eval()
        test_loss = 0
        test_correct = 0
        length_test = 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(test_loader)):
                original, translated, rotated, label = data
                gt = original[0].to(args.device)
                label = label.to(args.device).squeeze()
                if args.loss_net == 'PointNet':
                    output, _, _, _ = model(gt)
                else:
                    output, _ = model(gt)

                loss = criterion(output, label)
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                test_correct += (pred == label).sum().item()
                length_test += 1

        print("epoch: {}, train loss: {}, train accuracy: {}, test loss: {}, test accuracy: {}".format(
            epoch, train_loss / length_train, train_correct / length_train, test_loss / length_test, test_correct / length_test))

        if test_correct / length_test > best_acc:
            best_acc = test_correct / length_test
            best_epoch = epoch
            torch.save(model.state_dict(), args.exp+'/loss_network_'+str(num_points)+'.pth')

    print("Best accuracy: {}, at epoch: {}".format(best_acc, best_epoch))


def train_completion_network(model, loss1, loss2, args):
    if args.dataset == 'modelnet40':
        train_path = os.path.join(DATA_PATH, 'modelnet40_ply_hdf5_2048/train_files.txt')
        test_path = os.path.join(DATA_PATH, 'modelnet40_ply_hdf5_2048/test_files.txt')
        train_set = ModelNet40(num_points=2048, data_path=train_path, augmentation=args.augmentation)
        test_set = ModelNet40(num_points=2048, data_path=test_path, augmentation=args.augmentation)
    else:
        pass
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
            if args.augmentation:
                original, translated, rotated, _ = data
            else:
                original, _ = data

            # original data
            cutout = original[2].to(args.device)
            optimizer.zero_grad()
            coarse_pred, dense_pred = model(cutout)

            cd_1 = cd_loss_L1(coarse_pred, original[1].to(args.device))
            cd_2 = cd_loss_L1(dense_pred, original[0].to(args.device))
            if loss1 is not None and loss2 is not None:
                coarse_loss = loss1(coarse_pred, original[1].to(args.device))
                dense_loss = loss2(dense_pred, original[0].to(args.device))
                loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
            else:
                loss = cd_2 + alpha * cd_1
            loss.backward()
            optimizer.step()

            if args.augmentation:
                cutout = translated[2].to(args.device)
                optimizer.zero_grad()
                coarse_pred, dense_pred = model(cutout)

                cd_1 = cd_loss_L1(coarse_pred, translated[1].to(args.device))
                cd_2 = cd_loss_L1(dense_pred, translated[0].to(args.device))
                if loss1 is not None and loss2 is not None:
                    coarse_loss = loss1(coarse_pred, translated[1].to(args.device))
                    dense_loss = loss2(dense_pred, translated[0].to(args.device))
                    loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
                else:
                    loss = cd_2 + alpha * cd_1
                loss.backward()
                optimizer.step()

                cutout = rotated[2].to(args.device)
                optimizer.zero_grad()
                coarse_pred, dense_pred = model(cutout)

                cd_1 = cd_loss_L1(coarse_pred, rotated[1].to(args.device))
                cd_2 = cd_loss_L1(dense_pred, rotated[0].to(args.device))
                if loss1 is not None and loss2 is not None:
                    coarse_loss = loss1(coarse_pred, rotated[1].to(args.device))
                    dense_loss = loss2(dense_pred, rotated[0].to(args.device))
                    loss = 0.001*dense_loss+cd_2 + alpha * (0.001*coarse_loss + cd_1)
                else:
                    loss = cd_2 + alpha * cd_1

                loss.backward()
                optimizer.step()

            train_step += 1
        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(test_loader)):
                if args.augmentation:
                    original, translated, rotated, _ = data
                else:
                    original, _ = data
                cutout = original[2].to(args.device)
                coarse_pred, dense_pred = model(cutout)
                test_loss += l1_cd(dense_pred, original[0].to(args.device))
            print("epoch {}, L1 CD: {}".format(epoch, test_loss / len(test_loader.dataset)*1e3))
        if test_loss / len(test_loader.dataset) < best_l1_cd:
            best_l1_cd = test_loss / len(test_loader.dataset)
            best_epoch = epoch
            torch.save(model.state_dict(),
                       '/home/ethan/Code/Project/CMPUT414/model/PNLoss_alpha_2_PN(AU)_S5/completion_model_{}.pth'.format(epoch))
    print("Best L1 CD: {}, at epoch: {}".format(best_l1_cd, best_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='modelnet40',
                        choices=['modelnet40', 'shapenet'], help='dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=400, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_model', action='store_true', help='For Saving the current Model')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--augmentation', action='store_true', help='For data augmentation')
    parser.add_argument('--loss_net', type=str, default='PointNet',
                        choices=['PointNet', 'DGCNN', 'None'], help='loss network')
    parser.add_argument('--exp', type=str, default='', help='experiment name')
    args = parser.parse_args()

    # set up seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # set up experiment name
    if args.exp == '':
        args.exp = 'PCN_'+args.loss_net+'_'+args.dataset+'_'+str(args.epochs)+'_'+str(args.seed)
    if args.augmentation:
        args.exp += '_augmentation'
    args.exp = 'PCN_alpha_2_PN(AU)_S5'+args.dataset+'_'+str(args.epochs)+'_'+str(args.seed)
    print("Experiment: {}".format(args.exp))

    loss_net_dense = PointNetCls(40).to(args.device)
    loss_net_coarse = PointNetCls(40).to(args.device)

    loss_net_dense.load_state_dict(torch.load('/home/ethan/Code/Project/CMPUT414/model/PN_2048_AU.pth'))
    loss_net_coarse.load_state_dict(torch.load('/home/ethan/Code/Project/CMPUT414/model/PN_128_AU.pth'))
    dense_loss = PointNetLoss(loss_net_dense, args.device)
    coarse_loss = PointNetLoss(loss_net_coarse, args.device)
    completion_network = PCN(num_dense=2048).to(args.device)
    # dense_loss = None
    # coarse_loss = None
    train_completion_network(completion_network, coarse_loss, dense_loss, args)
    print("Training completed")

    # set up or train loss network (PointNet or DGCNN)
    # if args.loss_net == 'PointNet':
    #     if args.dataset == 'modelnet40':
    #         loss_net_dense = PointNetCls(k=40).to(args.device)
    #         loss_net_coarse = PointNetCls(k=40).to(args.device)

    #         # if not exist pretrained model, train from scratch
    #         if not os.path.exists(os.path.join(MODEL_PATH, 'PN_2048{}.pth'.format('_AU' if args.augmentation else ''))):
    #             train_loss_modelnet40(loss_net_dense, 2048, args)
    #         else:
    #             loss_net_dense.load_state_dict(torch.load(os.path.join(
    #                 MODEL_PATH, 'PN_2048{}.pth'.format('_AU' if args.augmentation else ''))))

    #         if not os.path.exists(os.path.join(MODEL_PATH, 'PN_128{}.pth'.format('_AU' if args.augmentation else ''))):
    #             train_loss_modelnet40(loss_net_coarse, 128, args)
    #         else:
    #             loss_net_coarse.load_state_dict(torch.load(os.path.join(
    #                 MODEL_PATH, 'PN_128{}.pth'.format('_AU' if args.augmentation else ''))))

    #         dense_loss = PointNetLoss(loss_net_dense, args.device)
    #         coarse_loss = PointNetLoss(loss_net_coarse, args.device)

    #     elif args.dataset == 'shapenet':
    #         loss_net_dense = ...
    #         loss_net_coarse = ...

    #         # if not exist pretrained model, train from scratch
    #         if not os.path.exists(os.path.join('')):
    #             train_loss_modelnet40()
    #         else:
    #             loss_net_dense.load_state_dict(torch.load(os.path.join('')))

    #         if not os.path.exists(os.path.join('')):
    #             train_loss_modelnet40()
    #         else:
    #             loss_net_coarse.load_state_dict(torch.load(os.path.join('')))

    #         dense_loss = PointNetLoss(loss_net_dense, args.device)
    #         coarse_loss = PointNetLoss(loss_net_coarse, args.device)

    # elif args.loss_net == 'DGCNN':
    #     if args.dataset == 'modelnet40':
    #         loss_net_dense = DGCNN(40, 1024, 0.5)
    #         loss_net_coarse = DGCNN(40, 1024, 0.5)

    #         # if not exist pretrained model, train from scratch
    #         if not os.path.exists(os.path.join(MODEL_PATH, 'DGCNN_2048{}.pth'.format('_AU' if args.augmentation else ''))):
    #             train_loss_modelnet40(loss_net_dense, 2048, args)
    #         else:
    #             loss_net_dense.load_state_dict(torch.load(os.path.join(
    #                 MODEL_PATH, 'DGCNN_2048{}.pth'.format('_AU' if args.augmentation else ''))))

    #         if not os.path.exists(os.path.join(MODEL_PATH, 'DGCNN_128{}.pth'.format('_AU' if args.augmentation else ''))):
    #             train_loss_modelnet40(loss_net_coarse, 128, args)
    #         else:
    #             loss_net_coarse.load_state_dict(torch.load(os.path.join(
    #                 MODEL_PATH, 'DGCNN_2048{}.pth'.format('_AU' if args.augmentation else ''))))

    #         dense_loss = DGCNNLoss(loss_net_dense, args.device)
    #         coarse_loss = DGCNNLoss(loss_net_coarse, args.device)

    #     elif args.dataset == 'shapenet':
    #         loss_net_dense = ...
    #         loss_net_coarse = ...

    #         # if not exist pretrained model, train from scratch
    #         if not os.path.exists(os.path.join('')):
    #             train_loss_modelnet40()
    #         else:
    #             loss_net_dense.load_state_dict(torch.load(os.path.join('')))

    #         if not os.path.exists(os.path.join('')):
    #             train_loss_modelnet40()
    #         else:
    #             loss_net_coarse.load_state_dict(torch.load(os.path.join('')))

    #         dense_loss = DGCNNLoss(loss_net_dense, args.device)
    #         coarse_loss = DGCNNLoss(loss_net_coarse, args.device)

    # elif args.loss_net == 'None':
    #     dense_loss = None
    #     coarse_loss = None

    # print("Loss network: {}".format(args.loss_net))

    # # set up completion model
    # if args.dataset == 'modelnet40':
    #     completion_network = PCN(num_dense=2048).to(args.device)
    # else:
    #     completion_network = PCN().to(args.device)

    # train_completion_network(completion_network, coarse_loss, dense_loss, args)
    # print("Training completed")
