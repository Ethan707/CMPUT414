'''
Author: Yuxi Chen
Date: 2022-03-15 17:49:50
LastEditTime: 2022-04-11 04:42:45
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

SHAPE = ["airplane",
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


def load_data(data_path):
    download()
    all_data = []
    all_label = []
    with open(data_path, 'r') as f:
        for h5_name in f.readlines():
            f = h5py.File(h5_name.strip(), 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def plot_point_cloud(data, label=None, save_path='./', save_file=False):
    # data: (N, 3)
    # label: (N,)
    # save_path: str
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], '.')
    if label is not None and save_file:
        file_path = save_path+str(label)+'.png'
        plt.savefig(file_path)
    plt.show()


class ModelNet40(Dataset):
    def __init__(self, num_points, data_path, num_coarse=128, severity=1):
        super().__init__()
        self.data, self.label = load_data(data_path)
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
        assert coarseCloud.shape == translatedCoarse.shape == rotatedCoarse.shape

        original = (pointCloud, coarseCloud, cutoutCloud)
        translated = (translatedCloud, translatedCoarse, translatedCutout)
        rotated = (rotatedCloud, rotatedCoarse, rotatedCutout)

        return original, translated, rotated, label

    def __len__(self):
        return self.data.shape[0]


# at most 2048 points for each sample
# total 9840 samples
# total 40 classes
if __name__ == '__main__':
    train = ModelNet40(1024, 'data/modelnet40_ply_hdf5_2048/train_files.txt', severity=5)

    # print("Translated:", example_data_translated.shape)
    # print("Noise:", example_data_noise.shape)
    # print("Rotated:", example_data_rotated.shape)
    # print("Label", example_label)

    original, translated, rotated, label = train[0]
    a0, a1, a2 = original
    b0, b1, b2 = translated
    c0, c1, c2 = rotated

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(a1[:, 0], a1[:, 1], a1[:, 2], '.')
    ax.scatter(b1[:, 0], b1[:, 1], b1[:, 2], '.')
    ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], '.')
    # ax.scatter(b[:, 0], b[:, 1], b[:, 2], '.', color='r', label='gt')

    plt.show()
    # label_set = {}
    # for _, data, label in test:
    #     assert len(label) == 1
    #     if label[0] not in label_set:
    #         label_set[label[0]] = 0
    #     label_set[label[0]] += 1
    # print(label_set)
