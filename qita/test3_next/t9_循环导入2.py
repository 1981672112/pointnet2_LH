from qita.test3_next.t7_循环导入1 import task1

def fun():
    print('循环导入2里面的func---1---')
    task1()
    print('循环导入2里面的func---2---')