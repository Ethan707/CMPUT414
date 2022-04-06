'''
Author: Yuxi Chen
Date: 2022-03-15 17:49:50
LastEditTime: 2022-04-06 11:48:13
LastEditors: Ethan Chen
Description:
FilePath: /CMPUT414/src/data.py
'''
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import h5py

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


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


def noise_pointcloud(pointcloud):
    # gaussian noise
    noise = np.random.normal(0, 0.005, size=pointcloud.shape)
    return pointcloud + noise


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
    def __init__(self, num_points, data_path='data/modelnet40_ply_hdf5_2048/train_files.txt', partition='train'):
        super().__init__()
        self.data, self.label = load_data(data_path)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, index):
        pointCloud = self.data[index][:self.num_points]
        ground_truth = pointCloud.copy()
        label = self.label[index]
        if self.partition == 'train':
            translatedCloud = translate_pointcloud(pointCloud)
            noiseCloud = noise_pointcloud(pointCloud)
            rotatedCloud = rotate_pointcloud(pointCloud)
            np.random.shuffle(pointCloud)
        return ground_truth, (translatedCloud, noiseCloud, rotatedCloud), label

    def __len__(self):
        return self.data.shape[0]


# at most 2048 points for each sample
# total 9840 samples
# total 40 classes
if __name__ == '__main__':
    train = ModelNet40(4096, 'data/modelnet40_ply_hdf5_2048/train_files.txt')
    test = ModelNet40(4096, 'data/modelnet40_ply_hdf5_2048/test_files.txt')
    example_data_gt, (example_data_translated, example_data_noise, example_data_rotated), example_label = train[0]
    print("Translated:", example_data_translated.shape)
    print("Noise:", example_data_noise.shape)
    print("Rotated:", example_data_rotated.shape)
    print("Label", example_label)
    # plot_point_cloud(example_data_translated, example_label, './images/')
    # plot_point_cloud(example_data_noise, example_label, './images/')
    # plot_point_cloud(example_data_rotated, example_label, './images/')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(example_data_gt[:, 0], example_data_gt[:, 1], example_data_gt[:, 2], '.', color='r', label='gt')
    ax.scatter(example_data_translated[:, 0], example_data_translated[:, 1],
               example_data_translated[:, 2], '.', color='b', label='translated')
    ax.scatter(example_data_noise[:, 0], example_data_noise[:, 1],
               example_data_noise[:, 2], '.', color='g', label='noise')
    ax.scatter(example_data_rotated[:, 0], example_data_rotated[:, 1],
               example_data_rotated[:, 2], '.', color='y', label='rotated')
    plt.show()
    # label_set = {}
    # for _, data, label in test:
    #     assert len(label) == 1
    #     if label[0] not in label_set:
    #         label_set[label[0]] = 0
    #     label_set[label[0]] += 1
    # print(label_set)
