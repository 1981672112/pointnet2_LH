"""
PyCharm 的解释器设置确实是个比较复杂的问题。
不过万变不离其宗，只要你理解以下几个概念和用法，任何 Python 环境相关的问题都可以自己排查：
    sys.executable
    sys.path
    pip list
    <模块>.__file__

sys.executable：
     该属性是一个字符串，在正常情况下，其值是当前运行的 Python 解释器对应的可执行程序所在的绝对路径。

sys.path指定模块搜索路径的列表。
    默认情况下，python导入文件或者模块，会在sys.path里找模块的路径。
    如果在当前搜索路径列表sys.path中找不到该模块的话，就会报错。
"""
import sys

a = sys.executable
print(type(a), a)
# <class 'str'> /home/lh/anaconda3/envs/pn37113/bin/python3.7
# '/home/lh/anaconda3/envs/pn37113/bin/python3.7'

b = sys.path
# print(type(b), b)  # <class 'list'>
"""
['/home/lh/下载/pycharm包/pycharm-community-2022.2.3/plugins/python-ce/helpers/pydev', 
'/home/lh/下载/pycharm包/pycharm-community-2022.2.3/plugins/python-ce/helpers/third_party/thriftpy', 
'/home/lh/下载/pycharm包/pycharm-community-2022.2.3/plugins/python-ce/helpers/pydev', 
'/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master', 
'/home/lh/anaconda3/envs/pn37113/lib/python37.zip', 
'/home/lh/anaconda3/envs/pn37113/lib/python3.7', 
'/home/lh/anaconda3/envs/pn37113/lib/python3.7/lib-dynload', 

'/home/lh/.local/lib/python3.7/site-packages', 
'/home/lh/.local/lib/python3.7/site-packages/chamfer-2.0.0-py3.7-linux-x86_64.egg',
 '/home/lh/.local/lib/python3.7/site-packages/emd_ext-0.0.0-py3.7-linux-x86_64.egg', 
 '/home/lh/anaconda3/envs/pn37113/lib/python3.7/site-packages', 
 '/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master']
"""

"""
当指定模块（或包）没有说明文档时，仅通过 help() 函数或者 __doc__ 属性，
无法有效帮助我们理解该模块（包）的具体功能。在这种情况下，
我们可以通过 __file__ 属性查找该模块（或包）文件所在的具体存储位置，直接查看其源代码。
"""

import string

print(string.__file__)  # 查看模块和包的具体位置 /home/lh/anaconda3/envs/pn37113/lib/python3.7/string.py
"""
仍以前面章节创建的 my_package 包为例，下面代码尝试使用 __file__ 属性获取该包的存储路径：
import my_package
print(my_package.__file__)
程序输出结果为：
C:\用户\mengma\Desktop\my_package\__init__.py
注意，因为当引入 my_package 包时，其实际上执行的是 __init__.py 文件，
因此这里查看 my_package 包的存储路径，输出的 __init__.py 文件的存储路径。
"""
