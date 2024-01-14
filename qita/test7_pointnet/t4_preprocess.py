import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import torch.nn as nn


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class Preprocess(Dataset):
    def __init__(self, root):
        self.root = root
        self.num_category = 40
        self.process_data = True
        self.save_path = '/home/lh/point_cloud/test1_pointnet2/pointnet2_LH/qita/test7_pointnet/data'
        self.uniform = True
        self.npoints = 1024
        split = 'train'

        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')  # 是一个路径 该文件里面是40个类别 每行一个类别
        self.cat = [line.rstrip() for line in open(self.catfile)]  # 把40个类别的字符串名字放在一个列表里面 40个元素的列表
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # 40类别的键值对 ‘data’：0，‘bathtub’：1，以此类推 共40个键值对

        shape_ids = {}  # 索引

        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]  # 所有train文件名 没有后缀
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]  # 所有test文件名 没有后缀
        # shape_ids是字典 两个键 对应 名称 一个txt是该形状所有点

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]  # 用列表装每个名称 对应的类别 split=train或者test
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]  # 拿到每个9843训练文件的绝对路径[(，),(，)] 9843个元组的列表 一个元组两个元素 类别和该类别的txt的绝对路径
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))  # 40类别 train 1024

        if self.process_data:  # F
            if not os.path.exists(self.save_path):  # 该路径真存在 不存在则进行下面的代码
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)  # [None, None]
                self.list_of_labels = [None] * len(self.datapath)  # [None, None]
                # self.classes 字典 40类别的键值对 ‘data’：0，‘bathtub’：1，以此类推 共40个键值对
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]  # index=0 o 元组 ('airplane', './data/airplane/airplane_0001.txt')
                    cls = self.classes[self.datapath[index][0]]  # cls=0
                    cls = np.array([cls]).astype(np.int32)  # ndarray [0]
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)  # ndarray 10000 6 dtype：float32
                    # 不加.astype(np.float32) point_set的dtype： float64

                    if self.uniform:  # fps均匀采样
                        point_set = farthest_point_sample(point_set, self.npoints)  # ndarray 1024 6
                    else:
                        point_set = point_set[0:self.npoints, :]  # 直接取前1024个点 1024 6

                    self.list_of_points[index] = point_set  # 列表 每个类别的样本的1024 6 的模型输入
                    self.list_of_labels[index] = cls  # 该样本对应的类别

                with open(self.save_path, 'wb') as f:  # self.save_path= './data/modelnet40_train_1024pts.dat'
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)


if __name__ == '__main__':
    p = Preprocess('./data')
    """
    cls = np.array([0]).astype(np.int32) 的意思是创建一个包含一个元素的NumPy数组，并将其数据类型(dtype)设置为np.int32。具体解释如下：
        np.array([0]): 创建一个包含单个元素 0 的NumPy数组。
        .astype(np.int32): 将数组的数据类型转换为32位整数（np.int32）。
    """
"""
point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
np.loadtxt(文件的绝对地址,delimiter=',') 还可以读取CSV文件 savetxt函数间数据存储到文件
.astype(np.float32)转换
"""
