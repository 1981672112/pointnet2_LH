"""
reversed 函数:
    reversed 函数返回一个反转的迭代器。
    语法:
        reversed(seq)
        参数:
        seq -- 要转换的序列，可以是 tuple, string, list 或 range。

"""
# 1 字符串
seqString = 'Runoob'
print(list(reversed(seqString)))  # <reversed object at 0x7f73df843390> 没有强制转换成列表

# 2 元组
seqTuple = ('R', 'u', 'n', 'o', 'o', 'b')
print(list(reversed(seqTuple)))

# 2 range
seqRange = range(5, 9)
print(list(reversed(seqRange)))

# 3 列表
seqList = [1, 2, 4, 3, 5]
print(list(reversed(seqList)))
"""
out:
    ['b', 'o', 'o', 'n', 'u', 'R']
    ['b', 'o', 'o', 'n', 'u', 'R']
    [8, 7, 6, 5]
    [5, 3, 4, 2, 1]
"""

"""
2 其中 yaml.safe_load 是 PyYAML 库中的一个函数，
用于将 YAML 格式的字符串转换为 Python 对象。
这段代码的作用可能是将多个 YAML 文件的内容合并到一个对象中。
"""

"""
3 如果指定的对象拥有指定的类型，则 isinstance() 函数返回 True，否则返回 False。
如果 type 参数是元组，则如果对象是元组中的类型之一，那么此函数将返回 True。
isinstance(object, type)
    object	必需。对象。
    type	类型或类，或类型和/或类的元组。
"""


class myObj:
    name = "Bill"


y = myObj()

x = isinstance(y, myObj)
print(x)  # True

"""
4 hasattr() 函数用于判断对象是否包含对应的属性。
hasattr 语法：
    hasattr(object, name)
    object -- 对象。
    name -- 字符串，属性名。
    如果对象有该属性返回 True，否则返回 False。

"""


class Coordinate:
    x = 10
    y = -5
    z = 0

    def __init__(self, name):
        self.name = name


point1 = Coordinate('1')
print(hasattr(point1, 'x'))  # T
print(hasattr(point1, 'y'))  # T
print(hasattr(point1, 'z'))  # T
print(hasattr(point1, 'name'))  # T 对象属性和类属性
print(hasattr(point1, 'no'))  # F 没有该属性

import sys
import numpy as np

print(sys.argv[1:])
default = '==SUPPRESS=='
# if default is not SUPPRESS:
#     print('T')

seed = np.random.randint(1, 5)
print(seed)
