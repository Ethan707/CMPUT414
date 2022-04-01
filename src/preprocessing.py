import os
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from os.path import dirname, abspath


class Ply:
    def __init__(self, filename, num):
        self.num = num
        f = open(filename, 'rb')
        self.plydata = PlyData.read(f).elements[0].data
        list_ply = [list(i) for i in self.plydata]
        self.np_plydata = np.array(list_ply)
        f.close()

    def find_min_max(self):
        min_max = []
        for i in range(3):
            min_ = np.min(self.np_plydata[:, i])
            max_ = np.max(self.np_plydata[:, i])
            min_max.append([min_, max_])
        min_max = np.abs(np.array(min_max).reshape(-1))
        max_value = np.max(min_max)
        return [-max_value, max_value]

    def create_box(self):
        low, high = self.find_min_max()
        box = np.linspace(low, high, self.num)
        box_buffer = np.zeros((self.num, self.num, self.num))
        for i in range(self.np_plydata.shape[0]):
            x, y, z = self.np_plydata[i]
            record = []
            for each_axis in [x, y, z]:
                for j in range(self.num - 1):
                    if box[j] <= each_axis <= box[j + 1]:
                        record.append(j)
                        break
            # print("\n\n\n")
            # print(record)
            # print(low, high)
            # print([x, y, z])
            box_buffer[record[0], record[1], record[2]] += 1
        return box_buffer

    def draw_box(self):
        box = self.create_box()
        x, y, z = np.where(box > 0)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        plt.show()


def upsampling(data):
    pass


if __name__ == '__main__':
    dataset_path = dirname(dirname(dirname(abspath(__file__)))) + "/dataset/"
    given_dataset_path = dataset_path + "shape_net_core_uniform_samples_2048/"
    all_folders = os.listdir(given_dataset_path)
    for i in all_folders:
        full_path = given_dataset_path + i + "/"
        for j in os.listdir(full_path):
            file_path = full_path + j
            print(file_path)
            single_ply = Ply(file_path, 32)
            box = single_ply.create_box()
            
            np.save(dataset_path + "data/" + j[:-3] + "npy", box)

    # single_ply = Ply("sample.ply", 32)
    # box = single_ply.create_box()
    # single_ply.draw_box()
