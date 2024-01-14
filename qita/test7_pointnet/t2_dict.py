"""
1 dict()
"""
# a = dict()  # 创建空字典
# a1 = dict(a='a', b='b', t='t', s=123)  # 传入关键字
# a2 = dict(zip(['one', 'two', 'three'], [1, 2, 3]))  # 映射函数 方式来构造字典
# a3 = dict([('one', 1), ('two', 2), ('three', 3)])  # 可迭代对象 方式来构造字典
# print(a)  # {}
# print(a1)  # {'a': 'a', 'b': 'b', 't': 't', 's': 123}
# print(a2)  # {'one': 1, 'two': 2, 'three': 3}
# print(a3)  # {'one': 1, 'two': 2, 'three': 3}

"""
2 split()
split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
str.split(str="", num=string.count(str)).
    str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
    num -- 分割次数。默认为 -1, 即分隔所有。
    返回分割后的字符串列表。
"""
# str = "Chris_iven+Chris_jack+Chris_lusy"
# print(str.split("+"))  # 返回分割后的字符串列表。
# print(str.split("_"))
# ['Chris_iven', 'Chris_jack', 'Chris_lusy']
# ['Chris', 'iven+Chris', 'jack+Chris', 'lusy']

"""
3 str.join(元组、列表、字典、字符串) 之后生成的只能是字符串。
所以很多地方很多时候生成了元组、列表、字典后，可以用 join() 来转化为字符串。
如果列表中只有一个元素，连接的操作其实是不必要的，因为该元素已经是一个字符串。
"""
# list = ['1', '2', '3', '4', '5']
# list1 = ['123']
# print(''.join(list))  # 12345
# print('_'.join(list1))  # 123 如果列表中只有一个元素，连接的操作其实是不必要的，因为该元素已经是一个字符串。
# seq = {'hello': 'nihao', 'good': 2, 'boy': 3, 'doiido': 4}
# print('-'.join(seq))  # 字典只对键进行连接 hello-good-boy-doiido
