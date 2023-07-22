"""
5. 高阶：不带参数的类装饰器
以上都是基于函数实现的装饰器，在阅读别人代码时，还可以时常发现还有基于类实现的装饰器。

基于类装饰器的实现，必须实现 __call__ 和 __init__两个内置函数。
__init__ ：接收被装饰函数
__call__ ：实现装饰逻辑。
"""


# 还是以日志打印这个简单的例子为例


class logger(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("[INFO]: the function {func}() is running..." \
              .format(func=self.func.__name__))
        return self.func(*args, **kwargs)


@logger
def say(something):
    print("say {}!".format(something))


say("hello")


class timer(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("woshi 函数")
        r1 = self.func(*args, **kwargs)  # 这里实际执行函数

        return r1  # return给r2 实际执行函数给外部的返回值 看外部要不要 即把two(4)结果赋值给r2


@timer
def two(a):
    # print(a * 2)
    return a * 2  # return给r1 因为在这里调用的two函数


r2 = two(4)
print(r2)
