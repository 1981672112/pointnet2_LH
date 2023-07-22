"""
导入模块：
    sys.path.append("..") ： 可调用上一级的python文件或包。
    1 import 模块名
        模块.变量，函数，类
    2 from 模块名 import 变量函数类
        在代码中可以直接使用变量，函数，类
    3 from 模块名 import *
        _x和__xx不可以导入，这种用import XX.py 可以导入
            如果要导入_x和__xx 只需将_x和__xx放入__all__
        该模块中的所用内容(加入不想导入所用的)
        __all__=['',] 限制*拿到的东西 只拿拿里面的 在要导入的模块添加
    4 无论是import还是from形式，都将会将模块内容加载（执行一遍）原模块希望调用的不执行
        if __name__ == '__main__':
            把执行的函数放下面
        在自己模块里面__name__叫：__main__
        如果在其他模块中通过导入的方式用的话：__name__叫：模块名
"""


list1 = [1, 2, 3, 7, 8, 9]
# import t4_calculate
#
# # 1调用模块变量
# print(t4_calculate.number)
#
# # 2调用模块函数
# resual = t4_calculate.add(*list1)
# print(len(list1), *list1)
# print(resual)
#
# # 3调用模块类
# cal = t4_calculate.Calculate(88)
# cal.test()
#
# # 4调用模块类方法
# t4_calculate.Calculate.test1()


# from t4_calculate import add, number, Calculate
from t4_calculate import *

# from t4_calculate import number # 同一模块导入不用写两遍 加逗号 可以直接使用类，函数，变量名了


resual = add(*list1)
print(len(list1), *list1)
sum = resual + number
print(sum)

c = Calculate(80)
c.test()
