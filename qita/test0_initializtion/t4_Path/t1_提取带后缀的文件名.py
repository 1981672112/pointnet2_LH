from pathlib import Path

cwd = Path.cwd()
print(cwd)
path = Path(__file__)
print(path)

# 1.1 创建文件夹1
data_folder = cwd / 'data_folder'
if not data_folder.exists():
    data_folder.mkdir()

# 1.2 创建文件夹2
test = cwd / Path('hhh')
test.mkdir()
# if not test.exists():
#     test.mkdir()

"""
1.3
pathlib.Path('./data').mkdir(parents=True, exist_ok=True)
parents = True: 创建中间级父目录
exist_ok= True: 目标目录存在时不报错
"""

# 2 删除文件夹
# if data_folder.exists():
#     data_folder.rmdir() # 只能删除空文件夹

# 3 创建txt文本
epmty_file_path = data_folder / 'empty.txt'
epmty_file_path.touch()

# 4 删除文件
# if epmty_file_path.exists():
#     epmty_file_path.unlink()


# /home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test0_initializtion/t4_Path/t1_提取带后缀的文件名.py
# 获得当前文件的绝对路径

path_str = Path(r"/usr/HinGwenWoong/demo.py")
path_file_name = path_str.name
print(path_file_name)


"""
.cwd() .mkdir() .rmdir() .touch() .unlink()
1 .name 
2 .parent
3 .suffix
4 .stem
5 .with_suffix(".json")
6 .iterdir()
7 .joinpath("demo.py")
8 .is_absolute()
9 .is_dir()
10 .is_file()
11 .exists()
12 .glob('*.py')
"""