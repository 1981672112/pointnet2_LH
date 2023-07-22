"""
作为函数定义时：
    1、*参数收集所有未匹配的位置参数组成一个tuple对象，局部变量args指向此tuple对象
    2、**参数收集所有未匹配的关键字参数组成一个dict对象，局部变量kwargs指向此dict对象
    def temp(*args,**kwargs):
        pass

作为函数调用时：
    1、*参数用于解包tuple对象的每个元素，作为一个一个的位置参数传入到函数中
    2、**参数用于解包dict对象的每个元素，作为一个一个的关键字参数传入到函数中

my_tuple = ("wang","yuan","wai")
temp(*my_tuple)
#---等同于---#
temp("wangyuan","yuan","wai")

my_dict = {"name":"wangyuanwai","age":32}
temp(**my_dict)
#----等同于----#
temp(name="wangyuanwai",age=32)
"""


# 1 普通
def print_str1(first, second):
    print(first)
    print(second)


# print_str("hello")  # TypeError: print_str() missing 1 required positional argument: 'second'


# 2 可变位置参数*second
def print_str2(first, *second):
    """
    修改print_str（）函数可接受一个参数、也可接受数量不定的参数
    将print_str（）函数的最后一个参数修改为可变参数*second"
    """
    print(first)
    print(second)


# 2.1 传一个
print_str2("hello")
"""
hello
()
这次不再报错，传入的第一个字符串参数"hello"打印出来了，
没有传入参数的*second则打印的是一个tuple对象的字符串表示形式，即一个括号"()"  。 
注意:（）表示含有0个元素的tuple对象！
"""
# 2.2 传多个
print_str2("hello", "美女", "小猫", "青蛙")
"""
hello
('美女', '小猫', '青蛙')

第一个参数“hello”，正常打印在第一行……
第二个参数"美女"，第三个参数“小猫”，第四个参数“青蛙”在函数的内部被组装进1个新的tuple对象中，
而这个新的tuple对象会赋值给变量second，此时局部变量second指向了一个tuple对象

说明：函数调用时传入的参数，会按照从左到右的顺序依次在函数中使用，
最左侧的参数先由位置参数first使用（匹配），剩下的所有未匹配的参数会被自动收集到1个新的tuple对象中，
而局部变量second会指向这个新的tuple对象

注意：*参数只收集未匹配的位置参数
"""
# 2.3 直接传入一个*
numbers_strings = ("1", "2", "3", "4", "5")
print_str2(*numbers_strings)  # 注意这里的*numbers_strings
# 等价于print_str("1","2","3","4","5")
"""
1
('2', '3', '4', '5')

"""

# 2.4 未定义可变参数的函数被调用时，传入*参数会发生什么呢？
numbers_strings = ("1", "2")
print_str1(*numbers_strings)
# 等同于print_str1("1","2")

"""
def print_str1(first, second):
    print(first)
    print(second)
1
2
元组解包的过程中会将每一个元素依次放入到位置参数，这说明元组的解包功能的如下特点：
    1、可以在可变参数中使用
    2、也可以在未定义可变参数的函数上使用

元组解包功能是完全独立的一个功能
再次说明：*参数，出现在函数的不同的位置上时，具备不同的功能
    1、当*参数出现在函数定义时，表示可变参数
    2、当*参数出现在函数调用时，则表示解包功能
"""
