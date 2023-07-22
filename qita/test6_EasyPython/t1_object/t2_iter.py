"判断一个对象是否可迭代"
from collections.abc import Iterable, Iterator

'1 Iterable'
print(isinstance(90, Iterable))  # False
print(isinstance((), Iterable))  # True
print(isinstance([], Iterable))  # True
print(isinstance({}, Iterable))  # True
print(isinstance(set(), Iterable))  # True
print(isinstance('', Iterable))  # True

'2 Iterator'
print(isinstance(90, Iterator))  # False
print(isinstance((), Iterator))  # False
print(isinstance([], Iterator))  # False
print(isinstance({}, Iterator))  # False
print(isinstance(set(), Iterator))  # False
print(isinstance('', Iterator))  # False

'3 '
print('------------------------')
f = open('iter.data')  # iter.data is a filename
print(isinstance(f, Iterable))
print(isinstance(f, Iterator))

'4 for circulation'
lt = [1, 2, 3]
lt1 = iter(lt)
print(next(lt1))
print(next(lt1))
print(next(lt1))
# print(next(lt1))  # StopIteration
'In the fourth row rises an error'

'5 class '


class C:

    def __iter__(self):
        """只要有这个魔术方法python就认为是可迭代对象"""
        pass

    def __next__(self):
        """迭代器是在__iter__之上在实现一个魔术方法"""
        pass


c = C()
print(isinstance(c, Iterable))  # T 只有__iter__
# print(isinstance(c, Iterator))  # F 无__next__
print(isinstance(c, Iterator))  # T 有__next__
