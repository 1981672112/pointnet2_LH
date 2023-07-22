# func.py

import os


def load_txt():
    # filename = './test.txt'
    # return open(filename, 'r').read()

    modelu_path = os.path.dirname(__file__)
    filename = modelu_path + '/test.txt'
    return open(filename, 'r').read()


def get_cur_path1():
    import os
    print(os.path.abspath(os.curdir))
    return os.path.abspath(os.curdir)
    # os.getcwd()与os.curdir都是用于获取当前执行python文件的文件夹，
    # 不过当直接使用os.curdir时会返回‘.’(这个表示当前路径），
    # 记住返回的是当前执行python文件的文件夹，而不是python文件所在的文件夹。
    # 我在t1_RelativePath文件夹下面执行main.py 里面调用了该函数但是返回了
    # /home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test0_initializtion/t1_RelativePath


def get_cur_path3():
    import os
    print(os.path.dirname(__file__))
    return os.path.dirname(__file__)
    # /home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test0_initializtion/t1_RelativePath/dir2

# load_txt 上下注释都可以访问txt 但是下面的外部函数也可以访问到txt
# print(load_txt())
