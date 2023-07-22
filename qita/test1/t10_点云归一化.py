import numpy as np


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)  # 每列求和 》》 求x，y，z的平均值
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 移动变化后坐标 x，y，z各自平方，在求和 得每个点与平均值中心点的距离的平方 开方 得距离 max找最远的点
    pc = pc / m
    return pc


# 这是一个Python函数，它对用numpy数组表示的点云进行了一种称为“点云归一化”的归一化操作。
"""
该函数的具体步骤如下：
    求出点云数组pc在每列上的平均值，得到一个长度为3的数组centroid，分别表示x、y、z三个方向上的平均值。
    将点云数组pc中的每个点都减去centroid，实现将点云平移到以原点为中心的位置上。
    计算点云数组pc中的所有点到原点的距离的最大值m，作为缩放系数。
    将点云数组pc中的每个点都除以缩放系数m，实现对点云进行缩放，使得点云的所有点到原点的距离不超过1。
    最后返回归一化后的点云数组pc。
    
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  
    # 移动变化后坐标 x，y，z各自平方，在求和 得每个点与平均值中心点的距离的平方 开方 得距离 max找最远的点
    pc = pc / m 归一化
"""
