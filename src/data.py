'''
Author: Yuxi Chen
Date: 2022-03-15 17:49:50
LastEditTime: 2022-04-16 15:04:13
LastEditors: Ethan Chen
Description:
FilePath: /CMPUT414/src/data.py
'''
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm import tqdm
from util import *
import open3d as o3d
import torch
import random
import loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

MODELNET40_TRAIN = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/train_files.txt'
MODELNET40_TEST = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/test_files.txt'
SHAPENET_DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data/PCN'
SHAPE_NAME = ["airplane",
              "bathtub",
              "bed",
              "bench",
              "bookshelf",
              "bottle",
              "bowl",
              "car",
              "chair",
              "cone",
              "cup",
              "curtain",
              "desk",
              "door",
              "dresser",
              "flower_pot",
              "glass_box",
              "guitar",
              "keyboard",
              "lamp",
              "laptop",
              "mantel",
              "monitor",
              "night_stand",
              "person",
              "piano",
              "plant",
              "radio",
              "range_hood",
              "sink",
              "sofa",
              "stairs",
              "stool",
              "table",
              "tent",
              "toilet",
              "tv_stand",
              "vase",
              "wardrobe",
              "xbox",
              ]


def download():
    BASIC_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASIC_DIR, '../data')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(data_path, shape=-1):
    download()
    all_data = []
    all_label = []
    with open(data_path, 'r') as f:
        for h5_name in f.readlines():
            f = h5py.File(h5_name.strip(), 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            if shape <= 39 and shape >= 0:
                shape_index = np.where(label == shape)
                all_data.append(data[shape_index[0], :, :])
                all_label.append(label[shape_index[0], :])
            elif shape == -1:
                all_data.append(data)
                all_label.append(label)
            else:
                print("shape not match, shape should be in [0, 39] or -1 for all shapes")
                assert False
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def plot_point_cloud(data):
    # plot the point cloud with pyplot
    # data: (N, 3)
    # label: (N,)
    # save_path: str
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], '.')
    plt.show()


def plot_point_cloud_one_view(pointclouds, file_name, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                              xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pointclouds))]
    fig = plt.figure(figsize=(len(pointclouds)*3*1.4, 3*1.4))
    elev = 30
    azim = -45
    for i, (pointclouds, size) in enumerate(zip(pointclouds, sizes)):
        color = pointclouds[:0]
        ax = fig.add_subplot(1, len(pointclouds), i+1, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.scatter(pointclouds[:, 0], pointclouds[:, 1], pointclouds[:, 2],
                   zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[i])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.suptitle(suptitle)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    plt.show()
    fig.savefig(file_name)
    plt.close()


class ShapeNet(Dataset):
    def __init__(self, dataroot, split, category, augmentation=False):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane": "02691156",  # plane
            "cabinet": "02933112",  # dresser
            "car": "02958343",
            "chair": "03001627",
            "lamp": "03636649",
            "sofa": "04256520",
            "table": "04379243",
            "vessel": "04530566",  # boat

            # alias for some seen categories
            "boat": "04530566",  # vessel
            "couch": "04256520",  # sofa
            "dresser": "02933112",  # cabinet
            "airplane": "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus": "02924116",
            "bed": "02818832",
            "bookshelf": "02871439",
            "bench": "02828884",
            "guitar": "03467517",
            "motorbike": "03790512",
            "skateboard": "04225987",
            "pistol": "03948459",
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.augmentation = augmentation

        self.partial_paths, self.complete_paths = self._load_data()

    def __getitem__(self, index):
        if self.split == 'train':
            partial_path = self.partial_paths[index].format(random.randint(0, 7))
        else:
            partial_path = self.partial_paths[index]
        complete_path = self.complete_paths[index]

        partial_pc = self.random_sample(self.read_point_cloud(partial_path), 2048)
        complete_pc = self.random_sample(self.read_point_cloud(complete_path), 16384)

        if self.augmentation:
            pass
        else:
            return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))

        partial_paths, complete_paths = list(), list()

        for line in lines:
            category, model_id = line.split('/')
            if self.split == 'train':
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '_{}.ply'))
            else:
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.ply'))
            complete_paths.append(os.path.join(self.dataroot, self.split, 'complete', category, model_id + '.ply'))

        return partial_paths, complete_paths

    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)

    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]


class ModelNet40(Dataset):
    def __init__(self, num_points, data_path, num_coarse=128, severity=5, shape=-1, augmentation=False):
        super().__init__()
        self.data, self.label = load_data(data_path, shape)
        self.num_points = num_points
        self.severity = severity
        self.num_coarse = num_coarse
        self.augmentation = augmentation

    def __getitem__(self, index):
        # the original data
        pointCloud = self.data[index][:self.num_points]
        coarseCloud = self.data[index][:self.num_coarse]
        label = self.label[index]

        # for the input of the network
        cutoutCloud = cutout(pointCloud.copy(), self.severity)

        # shuffle the data
        np.random.shuffle(pointCloud)
        np.random.shuffle(coarseCloud)
        np.random.shuffle(cutoutCloud)

        original = (pointCloud, coarseCloud, cutoutCloud)

        if self.augmentation:
            # for the data augmentation
            translatedCloud, translatedCoarse = translate_pointcloud(pointCloud.copy(), coarseCloud.copy())
            rotatedCloud, rotatedCoarse = rotation(pointCloud.copy(), coarseCloud.copy(), self.severity)
            translatedCutout = cutout(translatedCloud.copy(), self.severity)
            rotatedCutout = cutout(rotatedCloud.copy(), self.severity)
            np.random.shuffle(translatedCoarse)
            np.random.shuffle(rotatedCoarse)
            np.random.shuffle(translatedCloud)
            np.random.shuffle(rotatedCloud)
            np.random.shuffle(translatedCutout)
            np.random.shuffle(rotatedCutout)
            translated = (translatedCloud, translatedCoarse, translatedCutout)
            rotated = (rotatedCloud, rotatedCoarse, rotatedCutout)

            return original, translated, rotated, label
        else:
            return original, label

    def __len__(self):
        return self.data.shape[0]


# at most 2048 points for each sample
# total 9840 samples
# total 40 classes
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set = ModelNet40(2048, MODELNET40_TRAIN, augmentation=False)
    test_set = ModelNet40(2048, MODELNET40_TEST, augmentation=False)
    original, label = train_set[0]
    print(original[2].shape)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    # test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=8)
    # model = loss.DGCNN(40, 1024, 0.5).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, 100, eta_min=0.001)
    # criterion = nn.CrossEntropyLoss()
    # best_acc = 0
    # best_epoch = 0
    # for epoch in range(100):
    #     train_loss = 0
    #     train_correct = 0
    #     for _, data in enumerate(tqdm(train_loader)):
    #         original, label = data

    #         optimizer.zero_grad()
    #         pred, _ = model(original[0].to(device))
    #         label = label.to(device).squeeze()
    #         loss_value = criterion(pred, label)
    #         loss_value.backward()
    #         optimizer.step()
    #         train_loss += loss_value.item()
    #         pred = pred.argmax(dim=1)
    #         train_correct += (pred == label).sum().item()

    #     scheduler.step()
    #     model.eval()
    #     test_loss = 0
    #     test_correct = 0
    #     with torch.no_grad():
    #         for _, data in enumerate(tqdm(test_loader)):
    #             original,  label = data

    #             pred, _ = model(original[0].to(device))
    #             label = label.to(device).squeeze()
    #             loss_value = criterion(pred, label)
    #             test_loss += loss_value.item()
    #             pred = pred.argmax(dim=1)
    #             test_correct += (pred == label).sum().item()

    #     print("epoch: {}, train loss: {}, train accuracy: {}, test loss: {}, test accuracy: {}".format(
    #         epoch, train_loss/(1*len(train_loader.dataset)),
    #         train_correct / (1*len(train_loader.dataset)),
    #         test_loss / (1*len(test_loader.dataset)),
    #         test_correct / (1*len(test_loader.dataset))))

    #     if test_correct / (1*len(test_loader.dataset)) > best_acc:
    #         best_acc = test_correct / (1*len(test_loader.dataset))
    #         best_epoch = epoch
    #         torch.save(model.state_dict(), "/home/ethan/Code/Project/CMPUT414/model/DGCNN_128.pth")
    # print("Best accuracy: {}, at epoch: {}".format(best_acc, best_epoch))

    # w = ModelNet40(2048, TEST_DATA_PATH, shape="airplane")
    # category_train = {}
    # category_test = {}

    # for i in range(len(s)):
    #     original, translated, rotated, label = s[i]
    #     if label[0] not in category_train:
    #         category_train[label[0]] = 0
    #     category_train[label[0]] += 1

    # for i in range(len(w)):
    #     original, translated, rotated, label = w[i]
    #     if label[0] not in category_test:
    #         category_test[label[0]] = 0
    #     category_test[label[0]] += 1

    # print(len(s))
    # print(len(w))
    # print(category_train)
    # print(category_test)

# 9840
# 2468
# {30: 680, 27: 104, 29: 128, 22: 465, 7: 197, 28: 115, 0: 625, 20: 149, 35: 344, 32: 90, 34: 163, 26: 239, 5: 335, 12: 200, 10: 79, 21: 284, 14: 200, 36: 267, 2: 515, 33: 392,
#     24: 88, 9: 167, 4: 572, 31: 124, 18: 145, 11: 137, 16: 171, 17: 155, 8: 889, 25: 231, 37: 475, 23: 200, 15: 149, 6: 64, 3: 173, 38: 87, 1: 106, 13: 109, 19: 124, 39: 103}
# {4: 100, 0: 100, 2: 100, 8: 100, 23: 86, 37: 100, 35: 100, 28: 100, 16: 100, 20: 20, 33: 100, 30: 100, 26: 100, 1: 50, 17: 100, 22: 100, 21: 100, 7: 100, 12: 86,
#     36: 100, 25: 100, 5: 100, 14: 86, 39: 20, 38: 20, 27: 20, 9: 20, 34: 20, 15: 20, 3: 20, 6: 20, 13: 20, 19: 20, 24: 20, 11: 20, 18: 20, 29: 20, 31: 20, 10: 20, 32: 20}
