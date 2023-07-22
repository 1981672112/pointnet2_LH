# 文件夹 包
# 文件夹：非py文件 包：py文件
# 文件夹变包：文件目录下放一个__init__.py文件就行

# 一个包可以存放多个模块
# 项目》包》模块》类 函数 变量


"1"

# package.py文件和包user和article是平级的
# 使用包中模块中的User类

# from user import models
#
# u = models.User('admin', '123456')
# u.show()

"2"
# models.User 类这么做还是别扭
# 》》
# from user.models import User
#
# u = User('admin', '123456')
# u.show()
#
# from article.models import Article
#
# a = Article("个人总结", "家伟")
# a.show()

"""
    article
        --__init__.py
        --models.py
        --....
    user
        --__init__.py
        --models.py
        --....
    
    package.py
    同目录下
    
from 包 import 模块
from 包.模块 import 类 函数 变量
from 包.模块 import *


"""

# from user.models import *
#
# print(version)
# u = User('admin', '123456')
# from user.models import version # 可以拿到version
