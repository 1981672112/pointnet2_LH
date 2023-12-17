import numpy as np

"""
dot()函数可以通过NumPy库调用，也可以由数组实例对象进行调用。
例如：a.dot(b) 与 np.dot(a,b)效果相同。
但矩阵积计算不遵循交换律,np.dot(a,b) 和 np.dot(b,a) 得到的结果是不一样的。
"""

# 1 单个数
a = 2
b = 3
print(np.dot(a, b, out=None))  # 6 该函数的作用是获取两个元素a,b的乘积.

# 2 一维数组的dot函数运算
# a1 = np.array([1, 2])
# b1 = np.array([2, 3])
a1 = np.array((1, 2))
b1 = np.array((2, 3))
print(np.dot(a1, b1, out=None))  # 8 括号和列表包裹是一样的结果

# 3 二维的dot函数运算
a2 = np.array([[1, 2], [2, 3]])
b2 = np.array([[2, 3], [3, 4]])
print(np.dot(a2, b2, out=None))  # 有的像矩阵乘法 括号不行了，要换成列表

"""
[[ 8 11]
 [13 18]] 13在第二行第一列
[1, 2]  [2, 3]
[2, 3]  [3, 4]
8=行乘列=1*2+2*3=2+6
13=2*2+3*3=4+9
"""
