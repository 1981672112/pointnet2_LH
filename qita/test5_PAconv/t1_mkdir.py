"""
Linux mkdir 命令
Linux mkdir（英文全拼：make directory）命令用于创建目录。

语法
mkdir [-p] dirName
    参数说明：
        -p 确保目录名称存在，不存在的就建一个。
实例
在工作目录下，建立一个名为 runoob 的子目录 :
mkdir runoob

在工作目录下的 runoob2 目录中，建立一个名为 test 的子目录。
若 runoob2 目录原本不存在，则建立一个。
（注：本例若不加 -p 参数，且原本 runoob2 目录不存在，则产生错误。）

mkdir -p runoob2/test
"""



