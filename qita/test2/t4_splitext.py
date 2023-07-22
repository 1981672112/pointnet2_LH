"""
python中的os.path模块用法：
    dirname():用于去掉文件名，返回目录所在的路径
    basename():用于去掉目录的路径，只返回文件名
    join():用于将分离的各部分组合成一个路径名
"""

import os

path = '/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test2/t1_json.py'
# 返回字符串类型
t1 = os.path.dirname(path)  # '/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test2'
t2 = os.path.basename(path)  # 't1_json.py'
t3 = os.path.join(t1, t2)  # '/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test2/t1_json.py'

# 返回元组类型
t4 = os.path.split(path)
# split():用于返回目录路径和文件名的元组
# ('/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test2', 't1_json.py')

t5 = os.path.splitext(path)
# splitext()：用于返回文件名（带路径的）和扩展名元组
# ('/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test2/t1_json', '.py')

t6 = os.path.splitdrive(path)
# splitdrive():用于返回盘符和路径字符元组
# ('', '/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test2/t1_json.py')
