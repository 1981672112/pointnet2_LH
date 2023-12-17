"""
Python 字典(Dictionary) keys() 函数 以列表返回一个字典所有的键。
keys()方法语法：
    dict.keys()
    返回值:
        返回一个字典所有的键。

"""
car = {
    "brand": "Porsche",
    "model": "911",
    "year": 1963
}

x = car.keys()

print(x, type(x))  # dict_keys(['brand', 'model', 'year']) <class 'dict_keys'>
print(car, type(car))  # {'brand': 'Porsche', 'model': '911', 'year': 1963} <class 'dict'>
print(list(x)[0])  # brand

for i in x:
    print(i, type(i))
# brand <class 'str'>
# model <class 'str'>
# year <class 'str'>

for i in car:
    print(i, type(i))
# brand <class 'str'>
# model <class 'str'>
# year <class 'str'>


"""
TypeError: 'dict_keys' object is not subscriptable
原因：dict_keys(['no surfacing'，'flippers'])，返回的是一个 dict_keys 对象，不再是 list 类型，也不支持 index 索引
所以强制转成 list 类型即可使用
<class 'dict_keys'>和<class 'dict'> 都可以使用for
"""
from collections.abc import Iterable, Iterator

'1 Iterable'
print(isinstance(x, Iterable))  # True
print(isinstance(x, Iterator))  # False


class Feb:
    def __init__(self, length):  # length取决于用户想产生多长的斐波那契数列
        self.a = 0  # a 和 b 是数列初始值
        self.b = 1
        self.length = length

    def __iter__(self):  # 自身就是可迭代的，所以这个函数返回自身即可
        return self

    def __next__(self):  # 核心部分：如何处理每一个迭代的结果
        while self.length > 0:
            self.length -= 1
            self.a, self.b = self.a + self.b, self.a
            return self.a


if __name__ == '__main__':
    s = Feb(8)
    print(next(s))
    print(next(s))
    print(next(s))
    print(next(s))
    print(next(s))
    print(next(s))
    print(next(s))
    print(next(s))
    print(next(s))

