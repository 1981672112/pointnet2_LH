# import torch
# from numpy import *

import numpy as np

# for i in range(0, 5, 1):
#     print(i)
#
# for i in list(arange(0, 5, 0.1)):
#     print(i)

"""1 配合 
from numpy import * 
arange(0, 5, 0.1)

import numpy as np
np.arange(0, 5, 0.1)
range(0, 5,0.1) 步长不能为小数
"""

# 2 enumerate 和 zip
seg = ['1', '2', '3', '4']
seg1 = ['4', '3', '2', '1']
for i in enumerate(seg):
    print(i, type(i))
for i, j in enumerate(seg):
    print(i, j, type(i), type(j))

print('--------------------------------------------')
for i in zip(seg, seg1):
    print(i, type(i))
for i, j in zip(seg, seg1):
    print(i, j, type(i), type(j))
print(zip(seg, seg1))  # <zip object at 0x7f564e710b40>
print(list(zip(seg, seg1)))  # [('1', '4'), ('2', '3'), ('3', '2'), ('4', '1')]
"""
enumerate 一个接受 (0, '1') <class 'tuple'>
两个接受 0 1 <class 'int'> <class 'str'> 返回 下标值和对应元素值

zip就是两个可迭代序列 对应元素组合 一个一个返回元组类型(一个变量接受且使用for循环) 
还可以同时for循环两个可迭代的对象(两个变量接受) 返回两个值

可用list()变为列表 则每个元素是元组
可以放多个可迭代序列在zip函数中
"""

# def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
#     ''' batch_pc: BxNx3 '''
#     for b in range(batch_pc.shape[0]):
#         dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
#         drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
#         if len(drop_idx) > 0:
#             batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
#     return batch_pc
#
#
# bp = np.random.random(2 * 8 * 3).reshape(2, 8, 3)
# print(bp)
# t = np.random.random((bp.shape[1])) <= 0.5  # 返回数组 TFTF等
# print('t:', t, type(t))
# t1 = np.where(t)  # 返回元组 元组里面放的是数组 满足条件为True返回该元素的索引
# print('t1:', t1, type(t1))
# drop_idx = t1[0]  # 返回数组
# print('drop_idx:', drop_idx, type(drop_idx))
# if len(drop_idx) > 0:
#     bp[0, drop_idx, :] = bp[0, 0, :]  # set to the first point
# print(bp)
