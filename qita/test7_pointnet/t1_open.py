# [line.rstrip() for line in open('/home/lh/point_cloud/all_datasets/modelnet40_normal_resampled/modelnet40_shape_names.txt')]
#  rstrip() 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符。
"""
1 open()
2 rstrip() 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符。
     lstrip() 方法用于截掉字符串左边的空格或指定字符。str.lstrip([chars])
     Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        语法str.strip([chars]);
"""
# for line in open('/home/lh/point_cloud/all_datasets/modelnet40_normal_resampled/modelnet40_shape_names.txt'):
#     # print(line)
#     print(line.rstrip('\n'), type(line.rstrip()), len(line.rstrip()), type(line), len(line))
#     # airplane <class 'str'> 8 <class 'str'> 9

"""
zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
zip([iterable, ...])
返回元组列表。python3 要加list()转换
事实上，在 Python 3 中，为了节省空间，zip()返回的是一个tuple的迭代器，>> for a in zip() for a，b in zip() 
这也是我们为什么要调用list()将它强制转换成list的原因。
不过，Python 2中，它直接返回的就是一个列表了。

"""
cat = ['a', 'b', 'c']
cat_op = ['c', 'b', 'a']
a = zip(cat, range(len(cat)))
a1 = zip(cat, range(len(cat)), cat_op)  # 可以多个列表一起组合
print(a)
print(list(a1))

for i in zip(cat, range(len(cat))):
    print(i)

for i, j in zip(cat, range(len(cat))):
    print(i, j)

for i in zip(a):
    print(i)  # (('a', 0),) 因为a已经是返回的一个tuple的迭代器 所以会有两个元组括号，以及一个逗号，因为zip没有组合的对象了

# for i,j in zip(a):
#     print(i, j)
# ValueError: not enough values to unpack (expected 2, got 1)


classes = dict(zip(cat, range(len(cat))))
