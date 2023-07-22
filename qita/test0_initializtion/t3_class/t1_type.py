"""
1、所有的 Python 的用户定义类，都是 type 这个类的实例
    可能会让你惊讶，事实上，类本身不过是一个名为 type 类的实例。
    在 Python 的类型世界里，type 这个类就是造物的上帝。这可以在代码中验证：
"""


class MyClass:
    pass


instance = MyClass()

print(type(instance))  # <class '__main__.MyClass'>
print(MyClass)  # <class '__main__.MyClass'>
print(type(MyClass))  # <class 'type'>

"""
2、用户自定义类，只不过是 type 类的__call__运算符重载
    当我们定义一个类的语句结束时，真正发生的情况，是 Python 调用 type 的__call__运算符。简单来说，当你定义一个类时，写成下面这样时：
"""


class MyClass1:
    data = 1

# Python 真正执行的是下面这段代码：class = type(classname, superclasses, attributedict)
# 这里等号右边的type(classname, superclasses, attributedict)，就是 type 的__call__运算符重载，它会进一步调用：
# type.__new__(typeclass, classname, superclasses, attributedict)
# type.__init__(class, classname, superclasses, attributedict)
# 由此可见，正常的 MyClass 定义，和你手工去调用 type 运算符的结果是完全一样的。
