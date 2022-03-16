'''
Author: Yuxi Chen
Date: 2022-03-15 17:49:50
LastEditTime: 2022-03-16 10:28:11
LastEditors: Ethan Chen
Description: 
FilePath: \Project\src\data.py
'''
import os
from torch.utils.data import Dataset
import numpy as np
import h5py


def download():
    BASIC_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASIC_DIR)
    DATA_DIR = os.path.join(BASIC_DIR, '../data')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shape_net_core_uniform_samples_2048')):
        www = 'https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget -O %s %s' % (zipfile, www))
        os.system('mv %s %s' % (zipfile, DATA_DIR))
        os.system('unzip -d %s %s' % (DATA_DIR, os.path.join(DATA_DIR, zipfile)))
        os.system('rm %s' % (os.path.join(DATA_DIR, zipfile)))

    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(data_path):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    pass


class ModelNet40(Dataset):
    def __init__(self, num_points, data_path, partition='train'):
        super().__init__()
        self.data, self.label = load_data(data_path)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, index):
        pointCloud = self.data[index][:self.num_points]
        label = self.label[index]
        if self.partition == 'train':
            pointCloud = ...
            np.random.shuffle(pointCloud)
        return pointCloud, label

    def __len__(self):
        return self.data.shape[0]
