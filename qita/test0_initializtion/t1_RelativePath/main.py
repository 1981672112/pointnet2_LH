# main.py
from dir2 import func
import os, sys

# func.py的作用是提供load_txt()函数，读取同级目录下test.txt文件中的内容并返回。
# 获取当前工作目录和脚本文件所在目录
"""
1. 获取当前工作目录：
    $(pwd)
    英文意思为：print name of current/working directory
2. 获取脚本文件所在目录：
    $(cd `dirname $0`; pwd)
    先使用dirname获得脚本的父目录，然后cd进入，在执行pwd
"""

if __name__ == '__main__':
    # print(os.path.abspath(os.curdir))
    # print(func.load_txt())  # FileNotFoundError: [Errno 2] No such file or directory: './test.txt'

    """
    为什么会这样呢？这是因为在函数调用的过程中，
    当前路径.代表的是被执行的脚本文件的所在路径。
    在这个情况中，.表示的就是main.py的所在路径，
    所以load_txt()函数会在 dir2文件夹中寻找test.txt文件。main.py在dir1文件夹下
    """
    # 那么怎么样才能在函数调用的过程中
    # 保持相对路径的不变呢？

    """
    在以下的三个函数中，第一个和第二个是大部分教程中的解决办法，但是这样是错误的，
    因为第一个和第二个函数所获取的"当前文件路径"都是被执行的脚本文件的所在路径，
    只有第三个函数返回的当前文件路径才是真正的、该函数所在的脚本文件的所在路径
    """

    # def get_cur_path1():
    #     import os
    #     return os.path.abspath(os.curdir)
    #
    #
    # def get_cur_path2():
    #     import sys
    #     return sys.argv[0]
    #
    #
    # def get_cur_path3():
    #     import os
    #     return os.path.dirname(__file__)

    c = func.get_cur_path1()
    print(c, 'c:')
    d = func.get_cur_path3()
    print(d, 'd:')
