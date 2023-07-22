"""
4. 进阶：带参数的函数装饰器
回过头去看看上面的例子，装饰器是不能接收参数的。
其用法，只能适用于一些简单的场景。不传参的装饰器，只能对被装饰函数，执行固定逻辑。

装饰器本身是一个函数，做为一个函数，
如果不能传参，那这个函数的功能就会很受限，只能执行固定的逻辑。
这意味着，如果装饰器的逻辑代码的执行需要根据不同场景进行调整，
若不能传参的话，我们就要写两个装饰器，这显然是不合理的。

"""
# 比如我们要实现一个可以定时发送邮件的任务（一分钟发送一封），
# 定时进行时间同步的任务（一天同步一次），就可以自己实现一个 periodic_task （定时任务）的装饰器，
# 这个装饰器可以接收一个时间间隔的参数，间隔多长时间执行一次任务。

"""
那我们来自己创造一个伪场景，可以在装饰器里传入一个参数，
指明国籍，并在函数执行前，用自己国家的母语打一个招呼。
"""


# 那我们如果实现这个装饰器，让其可以实现 传参 呢？
# 会比较复杂，需要两层嵌套。
def say_hello(contry):  # contry 传装饰器的参数
    def wrapper(func):  # 传函数
        def deco(*args, **kwargs):  # 传函数的参数
            if contry == "china":
                print("你好!")
            elif contry == "america":
                print('hello.')
            else:
                return

            # 真正执行函数的地方
            func(*args, **kwargs)

        return deco

    return wrapper


# 小明，中国人
@say_hello("china")
def xiaoming():
    print('中国人')


# jack，美国人
@say_hello("america")
def jack():
    print('美国人')


# 执行
xiaoming()
print("------------")
jack()
