'''
Author: Ethan Chen
Date: 2022-04-08 07:53:02
LastEditTime: 2022-04-08 10:04:34
LastEditors: Ethan Chen
Description: 
FilePath: /CMPUT414/src/util.py
'''
import numpy as np


def normalize(new_pc):
    new_pc[:, 0] -= (np.max(new_pc[:, 0]) + np.min(new_pc[:, 0])) / 2
    new_pc[:, 1] -= (np.max(new_pc[:, 1]) + np.min(new_pc[:, 1])) / 2
    new_pc[:, 2] -= (np.max(new_pc[:, 2]) + np.min(new_pc[:, 2])) / 2
    leng_x, leng_y, leng_z = np.max(new_pc[:, 0]) - np.min(new_pc[:, 0]), np.max(new_pc[:, 1]) - \
        np.min(new_pc[:, 1]), np.max(new_pc[:, 2]) - np.min(new_pc[:, 2])
    if leng_x >= leng_y and leng_x >= leng_z:
        ratio = 2.0 / leng_x
    elif leng_y >= leng_x and leng_y >= leng_z:
        ratio = 2.0 / leng_y
    else:
        ratio = 2.0 / leng_z
    new_pc *= ratio
    return new_pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def rotation(pointcloud, severity: int):
    N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity-1]
    theta = np.random.uniform(c-2.5, c+2.5) * np.random.choice([-1, 1]) * np.pi / 180.
    gamma = np.random.uniform(c-2.5, c+2.5) * np.random.choice([-1, 1]) * np.pi / 180.
    beta = np.random.uniform(c-2.5, c+2.5) * np.random.choice([-1, 1]) * np.pi / 180.

    matrix_1 = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    matrix_2 = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
    matrix_3 = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])

    new_pc = np.matmul(pointcloud, matrix_1)
    new_pc = np.matmul(new_pc, matrix_2)
    new_pc = np.matmul(new_pc, matrix_3).astype('float32')

    return normalize(new_pc)


def uniform_noise(pointcloud, severity: int):
    # TODO
    N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
    jitter = np.random.uniform(-c, c, (N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return normalize(new_pc)


def cutout(pointcloud, severity: int):
    N, C = pointcloud.shape
    c = [(2, 30), (3, 30), (5, 30), (7, 30), (10, 30)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0], 1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # pointcloud[idx.squeeze()] = 0
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    # print(pointcloud.shape)
    return pointcloud
