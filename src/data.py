'''
Author: Yuxi Chen
Date: 2022-03-15 17:49:50
LastEditTime: 2022-04-12 19:15:59
LastEditors: Ethan Chen
Description:
FilePath: /CMPUT414/src/data.py
'''
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import h5py
from util import *
import open3d as o3d

TRAIN_DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/train_files.txt'
TEST_DATA_PATH = '/home/ethan/Code/Project/CMPUT414/data/modelnet40_ply_hdf5_2048/test_files.txt'
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


class ModelNet40(Dataset):
    def __init__(self, num_points, data_path, num_coarse=128, severity=1, shape=-1):
        super().__init__()
        self.data, self.label = load_data(data_path, shape)
        self.num_points = num_points
        self.severity = severity
        self.num_coarse = num_coarse

    def __getitem__(self, index):
        # the original data
        pointCloud = self.data[index][:self.num_points]
        coarseCloud = self.data[index][:self.num_coarse]
        label = self.label[index]

        # for the data augmentation
        translatedCloud, translatedCoarse = translate_pointcloud(pointCloud.copy(), coarseCloud.copy())
        rotatedCloud, rotatedCoarse = rotation(pointCloud.copy(), coarseCloud.copy(), self.severity)

        # for the input of the network
        cutoutCloud = cutout(pointCloud.copy(), self.severity)
        translatedCutout = cutout(translatedCloud.copy(), self.severity)
        rotatedCutout = cutout(rotatedCloud.copy(), self.severity)

        # shuffle the data
        np.random.shuffle(pointCloud)
        np.random.shuffle(coarseCloud)
        np.random.shuffle(translatedCoarse)
        np.random.shuffle(rotatedCoarse)
        np.random.shuffle(translatedCloud)
        np.random.shuffle(rotatedCloud)
        np.random.shuffle(cutoutCloud)
        np.random.shuffle(translatedCutout)
        np.random.shuffle(rotatedCutout)

        original = (pointCloud, coarseCloud, cutoutCloud)
        translated = (translatedCloud, translatedCoarse, translatedCutout)
        rotated = (rotatedCloud, rotatedCoarse, rotatedCutout)

        return original, translated, rotated, label

    def __len__(self):
        return self.data.shape[0]


# at most 2048 points for each sample
# total 9840 samples
# total 40 classes
if __name__ == "__main__":
    s = ModelNet40(2048, TRAIN_DATA_PATH, shape=0)
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
