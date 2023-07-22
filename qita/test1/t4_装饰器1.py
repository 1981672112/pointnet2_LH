import time


def baiyu():
    print("我是攻城狮白玉")
    time.sleep(2)


def count_time(func):
    def wrapper():
        t1 = time.time()
        func()
        print("执行时间为：", time.time() - t1)

    return wrapper


def a():
    print("woshi a")


def decorate(func):
    def zhuangshi():
        print("开始装饰")
        func()
        print("装饰结束")

    return zhuangshi


@decorate
def b():
    print("woshi b")


if __name__ == '__main__':
    """ 
    baiyu = count_time(baiyu)

    """
    # baiyu = count_time(baiyu)  # 因为装饰器 count_time(baiyu) 返回的时函数对象 wrapper，这条语句相当于  baiyu = wrapper
    # baiyu()  # 执行baiyu()就相当于执行wrapper()

    a = decorate(a)
    a()

    b()
