"""
3. 入门：时间计时器
再来看看 时间计时器 实现功能：顾名思义，就是计算一个函数的执行时长。
"""


# 这是装饰函数
def timer(func):
    def wrapper(*args, **kw):
        t1 = time.time()
        # 这是函数真正执行的地方
        func(*args, **kw)
        t2 = time.time()

        # 计算下时长
        cost_time = t2 - t1
        print("{}花费时间：{}秒".format(func.__name__, cost_time))

    return wrapper


# 假如，我们的函数是要睡眠10秒。这样也能更好的看出这个计算时长到底靠不靠谱。

import time


@timer
def want_sleep(sleep_time):
    time.sleep(sleep_time)


want_sleep(0.5)
