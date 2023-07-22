"""
6. 高阶：带参数的类装饰器
上面不带参数的例子，你发现没有，只能打印INFO级别的日志，
正常情况下，我们还需要打印DEBUG WARNING等级别的日志。
这就需要给类装饰器传入参数，给这个函数指定级别了。

带参数和不带参数的类装饰器有很大的不同。
__init__ ：不再接收被装饰函数，而是接收传入参数。
 __call__ ：接收被装饰函数，实现装饰逻辑。
"""


class logger(object):
    def __init__(self, level='INFO'):
        self.level = level

    def __call__(self, func):  # 接受函数
        def wrapper(*args, **kwargs):
            print("[{level}]: the function {func}() is running..." \
                  .format(level=self.level, func=func.__name__))  # {变量}里面的变量 在format()里面以关键字方式传入 变量=某某
            func(*args, **kwargs)

        return wrapper  # 返回函数


@logger(level='WARNING')
def say(something):
    print("say {}!".format(something))


say("hello")


class CLanguage:
    def __init__(self):
        self.name = "C语言中文网"
        self.add = "http://c.biancheng.net"

    def say(self):
        print("我正在学Python")


clangs = CLanguage()
if hasattr(clangs, "name"):
    print(hasattr(clangs.name, "__call__"))
print("**********")
if hasattr(clangs, "say"):
    print(hasattr(clangs.say, "__call__"))
