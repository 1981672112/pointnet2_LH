# __all__=['number',]

number = 100
name = "calculate"


def add(*args):
    if len(args) > 1:
        sum = 0
        for i in args:
            sum += i
        return sum
    else:
        print("re-input")


class Calculate:
    def __init__(self, num):
        self.num = num

    def test(self):  # 对象方法
        print('using calculate')

    @classmethod
    def test1(cls):  #
        print("calculate类中的类方法")


def test():
    print("woshi ceshi")


print(__name__)

if __name__ == '__main__':
    # print(__name__)  # __main__
    test()
