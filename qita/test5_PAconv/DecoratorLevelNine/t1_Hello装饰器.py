"""
1. Hello,装饰器
装饰器的使用方法很固定:
    先定义一个装饰器（帽子）
    再定义你的业务函数或者类（人）
    最后把这装饰器（帽子）扣在这个函数（人）头上
"""


def decorator(func):
    def wrapper(*args, **kw):
        print('>>>>>')
        return func()

    return wrapper


@decorator
def function():
    print("hello, decorator")


function()
