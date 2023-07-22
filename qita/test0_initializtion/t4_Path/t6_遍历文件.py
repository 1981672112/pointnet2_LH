from pathlib import Path

path_str = Path(r"/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test0_initializtion/t4_Path")


for path in path_str.iterdir():
    print(path)
