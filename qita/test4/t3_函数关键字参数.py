def printStr(**anything):
    print(anything)


printStr(first=5, second=100)
# {'first': 5, 'second': 100}
"""
打印结果为dict对象的字符串形式，为什么anything成为dict了？

说明：函数调用时，传入的关键字参数有匹配的位置参数时，则位置参数优先使用（匹配）这些关键字参数，
剩余所有未使用（未匹配）的关键字参数会在函数内组装进一个dict对象中，组装后dict对象会赋值给变量名anything，此时局部变量anything指向一个dict对象

注意：**参数只收集未匹配的关键字参数
函数调用时使用字典解包功能（dict对象前加**）

"""


def printStr1(first, **dict):
    print(str(first) + "\n")
    print(dict)


printDic = {"name": "tyson", "age": "99"}

printStr1(100, **printDic)
# printStr(100, name = "tyson", age = "99")
# 说明：函数调用时，在一个dict对象的前面，添加**，表示字典的解包，
# 它会把dict对象中的每个键值对元素，依次转换为一个一个的关键字参数传入到函数中


"""
解包功能不只是tuple、还有list、str、range
只要是序列类型，可迭代对象，都可以使用解包功能哦
"""
# 比如说这个又6个数据的元组，我们感兴趣的只有开头和结尾的数据，不想要中间的数据，那需要像上面那样弄6个参数来接受吗？
tup = (1, 2, 3, 4, 5, 6)
a, *_, b = tup  # 中间的数据被打包到了 _ 中，我们只获取到了开头和结尾的数据

first = (1, 2, 3)
second = [1, 2, 3]
third = "123"
fourth = range(4)

print(*first, )
print(*second)
print(*third)  # 字符串解包成一个一个字母
print(*fourth)

# for i in first:
#     print(type(i))

print(type(first))
