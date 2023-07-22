# 1. Numpy：np.random.random()函数
import numpy as np

k = np.random.random((3, 4))
print(k)
"""
生成3行4列的浮点数，浮点数都是从0-1中随机,维度：2
"""

# 2. 处理数据时经常需要从数组中随机抽取元素，这时候就需要用到np.random.choice()。
"""
numpy.random.choice(a, size=None, replace=True, p=None)
从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
replace:True表示可以取相同数字，False表示不可以取相同数字
数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
"""
# 2.1 产生随机数
a0 = np.random.choice(5)  # 从[0, 5)中随机输出一个随机数 4
# 相当于np.random.randint(0, 5)

b = np.random.choice(5, 3)  # 在[0, 5)内输出五个数字并组成一维数组（ndarray） 结果：array([1, 4, 1])
# 相当于np.random.randint(0, 5, 3)

# 2.2 从数组、列表或元组中随机抽取 注意：不管是什么，它必须是一维的！
L = [1, 2, 3, 4, 5]  # list列表
T = (2, 4, 6, 2)  # tuple元组
A = np.array([4, 2, 1])  # numpy,array数组,必须是一维的
A0 = np.arange(10).reshape(2, 5)  # 二维数组会报错

# 控制台执行
np.random.choice(L, 5)  # array([1, 4, 1, 3, 1])
np.random.choice(T, 5)  # array([2, 6, 6, 2, 6])
np.random.choice(A, 5)
# np.random.choice(A0, 5)  # 如果是二维数组，会报错

# numpy.random.choice(a, size=None, replace=True, p=None)
"""
从a中，按照size的shape提取样本。
其中a若是array，则抽样元素，若是整数，则从np.arange(a)中抽样（即[0,a）)
replace参数的作用，是决定重复抽样还是不重复抽样。
当replace=True（默认模式）时，则为可重复抽样
当replace=False）时，则为不重复抽样
另外注意，想要不重复抽样，需要配合size参数使用，并在1次抽样中完成，而不能是多次（如使用for循环）见下面代码

"""
import numpy as np

a = np.arange(10)
print('a=', a)

print('可重复抽样：', np.random.choice(a, size=10))
print('不可重复抽样：', np.random.choice(a, size=10, replace=False))  # size>10时会报错

# 注意，下面的方式，产生的样本还可以是重复的，因为replace=False只能配合size参数在一次抽样中发挥作用，下面这种属于多次抽样
for i in range(10):
    print(np.random.choice(a, replace=False), end=',')
"""
result:
    a= [0 1 2 3 4 5 6 7 8 9]
    可重复抽样： [9 3 1 3 3 9 9 1 7 7]
    不可重复抽样： [1 7 2 9 3 0 8 5 4 6]
    2,1,9,8,7,3,2,1,4,4,
"""