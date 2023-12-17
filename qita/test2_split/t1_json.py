import os
import json

import warnings
import numpy as np
from torch.utils.data import Dataset

"""
shuffled_train_file_list.json 是一个列表 元素是字符串'shape_data/03624134/3d2cb9d291ec39dc58a42593b26221da'等等
在集合里面有12135个 其实有12137个数据 因为集合里面不能有重复的元素

shapenetcore_partanno_segmentation_benchmark_v0_normal里面 03624134/3d2cb9d291ec39dc58a42593b26221da.txt
"""

root = r"/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
a = []
with open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
    # for d in json.load(f):
    #     print(d, type(d))
    #     a.append(d)
    #     t3 = str(d.split('/'))  # "['shape_data', '03624134', '3d2cb9d291ec39dc58a42593b26221da']"
    #     t4 = str(d.split('/')[2])  # '3d2cb9d291ec39dc58a42593b26221da'

    # t1 = set([str(d.split('/')[2]) for d in json.load(f)]) # 会报错
    # train_ids = t1

    train_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # d='shape_data/03624134/3d2cb9d291ec39dc58a42593b26221da'
    print(type(train_ids), len(train_ids))

print(len(a))

# 文件目录结构 txt有2816行 7列 2753 7 1916 7
"""
shapenetcore_partanno_segmentation_benchmark_v0_normal
    02691156
        1a04e3eab45ca15dd86060f189eb133.txt
    02773838
    02954340
    02958343
    03001627
    03261776
    03467517
    03624134
    03636649
    03642806
    03790512
    03797390
    03948459
    04099429
    04225987
    04379243
    processed
    train_test_split
        shuffled_test_file_list.json
        shuffled_train_file_list.json
        shuffled_val_file_list.json
    synsetoffset2category.txt

"""
