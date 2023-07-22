"""
__init__.py文件 只要导包就默认执行这个文件

当导入包的时候，默认调用__init__.py文件
作用：
    1 当导入包的时候，把一些初始化的函数，变量，类定义在__init__文件中
    2 此文件中的函数，变量等的访问，只需要通过 包名.函数 即可访问
    3 结合__all__=[通过*可以访问的模块]

这里 直接导包没有问题（在包里模块导入要加包名 from 包名.模块名 import 类）
"""
import user

# from user.models import User

"使用上面两种导入都会出现：---------->user的__init__"
"user是包 下面可以访问到包__init__的函数 而不是user.__init__.printA"
user.craete_app()
user.printA()

# from 模块 import *
# user包名 这样却访问不了
# 应该加 在user包名 __all__=['models'] 才可以拿到models模块的User类
# 不加则什么也导入不进来
from user import *

user = models.User('admin', '123456')
user.show()

"models爆红，怎么找不到"


"""
关于__init__区别
    from 模块 import *  ‘变量，函数，类都行’
    from 包名 import *  '文件名'
"""