"""
os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
它不包括 . 和 .. 即使它在文件夹中。只支持在 Unix, Windows 下使用。
注意：针对目录下有中文目录对情况，Python2 需要经过编码处理，
但是在 Python3 中不需要已经没有 unicode() 方法，默认是 utf8 编码，所以需要转。

listdir()方法语法格式如下：
    os.listdir(path)
        path -- 需要列出的目录路径
    返回值:
        返回指定路径下的文件和文件夹列表。


sorted() 函数对所有可迭代的对象进行排序操作。
sort 与 sorted 区别：
    sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
    list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，
    而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。

sorted 语法：
    sorted(iterable, cmp=None, key=None, reverse=False)
    参数说明：
        iterable -- 可迭代对象。
        cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，
                此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
        key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，
                指定可迭代对象中的一个元素来进行排序。
        reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）123abc。
    返回值:
        返回重新排序的列表。
"""
import os

dir_point = '/home/lh/point_cloud/test1_pointnet2/' \
            'Pointnet_Pointnet2_pytorch-master/data'

a = os.listdir(dir_point)
print(a,)
print(type(a), len(a))

b = sorted(a)
print(b)
print(type(b), len(b))
