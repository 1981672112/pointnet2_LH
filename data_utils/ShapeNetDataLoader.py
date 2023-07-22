# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)  # 没列求和 》》 求x，y，z的平均值
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints  # 2048
        self.root = root  # '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/synsetoffset2category.txt'
        self.cat = {}
        self.normal_channel = normal_channel  # False

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()  # ['Airplane', '02691156']
                self.cat[ls[0]] = ls[1]  # {'Airplane': '02691156'} dict 16
        self.cat = {k: v for k, v in self.cat.items()}  # 字典 16个类对应的文件夹 {'Airplane': '02691156', ... , }
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))  # 类别对应0 1 2 例子
        # {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8,
        # 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        if not class_choice is None:  # 不做类别选择 把某些类给捞出来
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}  # 每个数据集打开的都不一样
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # 12135
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # 1870
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # 2873
        for item in self.cat:
            # print('category', item) # 在字典里面循环 循环的是键  item ='Airplane' 也就是说 item是类文件名
            self.meta[item] = []  # 字典 键=类 值=[]
            dir_point = os.path.join(self.root, self.cat[item])  # '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156'
            fns = sorted(os.listdir(dir_point))  # 飞机类 2690个txt文件 fns=['10aa040f470500c6a66ef8df4909ded9.txt',..., ]
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]  # 341个文件名 不要.txt 所以是-4 fn是fns的飞机类所有文件名带txt 选出在test_ids里面的文件 test_ids不带.txt
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:  # fn='10aa040f470500c6a66ef8df4909ded9.txt'
                token = (os.path.splitext(os.path.basename(fn))[0])  # '10aa040f470500c6a66ef8df4909ded9'
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))  # 一个飞机类的所有test 的具体路径被加进去 对应'Airplane'键 值是列表

        self.datapath = []  # [所有类 (飞机，一个飞机路径)，..,()]
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))  # 添加元组 飞机和路径 2874个 [('Airplane', '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/10aa040f470500c6a66ef8df4909ded9.txt'),.., ]

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]  # {'Airplane': 0} {'Airplane': 0, 'Bag': 1}...{ 共16 }

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple 从索引到 (point_set, cls, seg) 元组
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]  # ('Chair', '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03001627/b13a4df698183bf9afb6676a5cd782b6.txt')
            cat = self.datapath[index][0]  # 'Chair'
            cls = self.classes[cat]  # 4 int
            cls = np.array([cls]).astype(np.int32)  # [4] 数组ndarray
            data = np.loadtxt(fn[1]).astype(np.float32)  # ndarray 2746 7 读取txt文件 把点拿出来
            if not self.normal_channel:  # 没有法向量信息 》》true
                point_set = data[:, 0:3]  # point_set=点集 只拿前三维特征 所有xyz的信息
            else:
                point_set = data[:, 0:6]  # 有xyz信息 拿六维信息
            seg = data[:, -1].astype(np.int32)  # 数组 部件类别 2746个13 seg是一个txt的最后一列
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)  # self.cache是一个字典  点集xyz 4 物体类 13 物体每个点（2746）的类
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 把拿出来的点进行xyz归一化

        choice = np.random.choice(len(seg), self.npoints, replace=True)  # 这里采样2048个点 len(seg)长度可能比2048小，但是可以重复取
        # resample
        point_set = point_set[choice, :]  # 一个txt里取2048个点 [2048,3]
        seg = seg[choice]  # 新的2048个点 的数组

        return point_set, cls, seg  # 都是数组ndarray [2048 3] [15] [47 48 49]

    def __len__(self):
        return len(self.datapath)
