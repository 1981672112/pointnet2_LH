"""
循环导入：A中导B，B中导A
    A：模块
        def test():
            f()
    B: 模块
        def f():
            test()



"""

# from t9_循环导入2 import func

def task1():
    print('---task1---')


def task2():
    print('----task2---')
    fun()



task2()

fun1111()