"""
    article
        --__init__.py
        --models.py
        --....
    user
        --__init__.py
        --models.py
            --User
        --test.py
        --....
    问题1：test.py用同包models.py和；另外上一层的 article包的models.py
    问题2：test.py导入t5_package.py
"""
import sys

daoru = r'/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test3_next/article'
# daoru1 = r'/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test3_next/user'
# sys.path.insert(0, daoru)
# sys.path.insert(0, daoru1)

# sys.path.append(daoru)
# sys.path.remove('/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/qita/test3_next/user')
print(sys.path)

# 用户发表文章
# 创建用户对象

# 发表文章，文章对象

# import models  # 同包这样写会有红线，但是现在怎么没有 晕


from user.models import User

"也可以用下面的导入方法 . 代表当前目录 但是报错了"

# from .models import User
from article.models import Article

from qita.test3_next.t4_calculate import add

user = User('admin', '123456')
article = Article('个人总结', '家伟')

user.publish_article(article)

list1 = [1, 3, 5, 6, 7]
result = add(*list1)
print(result)
