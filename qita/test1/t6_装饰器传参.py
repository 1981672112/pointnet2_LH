"""
当我们的被装饰的函数是带参数的，此时要怎么写装饰器呢？
下面我们有定义了一个blog函数是带参数的
"""
import time


def blog(name):
    print('进入blog函数')
    name()
    print('我的博客是 https://blog.csdn.net/zhh763984017')


"此时我们的装饰器函数要优化一下下，修改成为可以接受任意参数的装饰器"


def count_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        print("执行时间为：", time.time() - t1)

    return wrapper


@count_time
def blog(name):
    print('进入blog函数')
    name()
    print('我的博客是 https://blog.csdn.net/zhh763984017')


def test():
    print("我是test")


if __name__ == '__main__':
    # baiyu = count_time(baiyu)  # 因为装饰器 count_time(baiyu) 返回的时函数对象 wrapper，这条语句相当于  baiyu = wrapper
    # baiyu()  # 执行baiyu()就相当于执行wrapper()

    # baiyu()  # 用语法糖之后，就可以直接调用该函数了
    blog(test)
