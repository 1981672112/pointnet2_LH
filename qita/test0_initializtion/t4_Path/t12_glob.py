from pathlib import Path

# path_str = Path(r"/user/HinGwenWoong/scripts")
path_str = Path(r"/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models")
print(path_str.glob('*.py'))
# <generator object Path.glob at 0x7f9e898b1750>
# f是一个迭代器对象，通过遍历，可以输出所有满足条件的*.py文件

for py in path_str.glob('*.py'):
    print(py)
"""
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_part_seg_ssg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/csa_for_seg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_utils_xiugai.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet_sem_seg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet_part_seg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_cls_ssg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_cls_msg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/csa_for_seg_gai.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_part_seg_msg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet_utils.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_sem_seg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_utils.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_sem_seg_msg.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/point_cloud_transformer_cls_zj.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_utils_xiugai_cross.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet2_part_seg_ssg_xiugai.py
/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/models/pointnet_cls.py
"""
