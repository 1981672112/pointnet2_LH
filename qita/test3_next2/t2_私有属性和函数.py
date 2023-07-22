"""
在Python中，使用一个下划线前缀来表示一个“内部使用”的变量或方法是一个约定，而不是一个强制性的规则。
这个约定告诉其他程序员，这个变量或方法不应该在类的外部使用。但是，如果需要，它们仍然可以被访问和使用。因此，它们并不是完全私有的。

对于一个真正私有的变量或方法，可以使用两个下划线前缀。这将使它们成为类的私有成员，不能在类的外部访问。
如果您确实需要从类的外部访问这些私有成员，可以提供公共接口方法来获取或设置它们的值。

"""
import torch
import torch.nn as nn

print(torch.version.cuda)


class Person(nn.Module):
    def __init__(self, age):
        super(Person, self).__init__()
        self.age = age
        self._age = age
        self.__age = age

    def speak_age(self):
        print('1个', self.age)
        print('2个', self._age)
        print('3个', self.__age)  # 内部可以调用私有属性
        return str(f'我不知道我的年龄，可能是{self.age}岁。')

    def _speak_age(self):
        print(self._age)

    def __speak_age(self):
        print(self._age)

    def forward(self, age):
        print("用一下私有方法")
        self.__speak_age()
        if self.age != age:
            print("你的年龄有问题")
        else:
            print("你是诚实孩子")


person = Person(8)

# 私有属性
# print(person.age)  # 8
# print(person._age)  # 8
# print(person.__age)  # AttributeError: 'Person' object has no attribute '__age'

print("-----------")
# 私有方法
# person.speak_age()  # 8
# person._speak_age()  # 8
# person.__speak_age()  # AttributeError: 'Person' object has no attribute '__speak_age'

# print("测试：print(person.speak_age())={}".format(print(person.speak_age())))
# print(person.speak_age())
"""
8
None
"""
person.speak_age()

input = 4
out = person(8)
"""
继承nn.Module是为了将自定义的类作为神经网络模块进行使用，这样可以使用PyTorch提供的许多功能来管理和优化模型，
例如自动求导、参数管理、模型保存和加载等。在继承nn.Module的类中，必须实现__init__和forward两个方法。
__init__方法用于定义模型的结构，forward方法用于描述模型的前向传递过程。
通过继承nn.Module，可以更加方便地定义和使用自己的神经网络模型。


super(Person, self).__init__() 是调用 nn.Module 类的构造函数，
这样可以确保在创建 Person 类实例时，也会调用 nn.Module 的构造函数。
通过这种方式，Person 类会继承 nn.Module 类的所有属性和方法，而不会覆盖它们。
在神经网络模型中，通常需要使用一些基础的属性和方法，例如参数管理、优化器、模型保存和加载等。
通过继承 nn.Module 类，可以更加方便地使用这些属性和方法。
在调用 super(Person, self).__init__() 后，可以在 Person 类中使用 self 对象访问所有 nn.Module 的属性和方法。
"""
