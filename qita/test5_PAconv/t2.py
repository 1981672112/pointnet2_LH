import time


# 装饰器fun3
def fun3(f):
    def inner(*args, **kwargs):
        print('inner>>>  ', end='')
        r = f(*args, **kwargs)
        return r

    return inner


@fun3
def fun1(n):
    for i in range(n):
        j = i * 2
    return j


@fun3
def fun2(n):
    s = ""
    for i in range(n):
        s += str(i)


# t = time.time()
# fun1(10)
# print(time.time() - t)
#
# t = time.time()
# fun2(10)
# print(time.time() - t)

# 怎么简化

# print(fun1)  # <function fun3.<locals>.inner at 0x7fac64fd6710>
# print(fun1(10))  # None
# fun3(fun1)(10)  # 等价fun1(10)
# inner(10)
# print(fun3(fun1)(10))  # NoneNone

"""
fun1不加装饰器fun3 
print(fun3(fun1)(10))
等价
fun1加装饰器fun3 
print(fun1(10))
"""
fun1(10)  # inner>>>
fun2(10)  # inner>>>
print(end='\n')
print(fun1(10))  # inner>>>  18 //fun1 have return
print(fun2(10))  # inner>>>  None //fun2 have no return


def decorator(func):
    def wrapper(*args, **kw):
        print('>>>>>')
        return func(*args,**kw)

    return wrapper


@decorator
def function():
    print("hello, DecoratorLevelNine")


function()
