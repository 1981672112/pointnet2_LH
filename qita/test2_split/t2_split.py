"""
split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
split() 方法语法：
        str.split(str="", num=string.count(str)).
        str -- 分隔符，默认为所有的 空字符，包括空格、换行(\n)、制表符(\t)等。
        num -- 分割次数。默认为 -1, 即分隔所有。
        返回值：返回分割后的字符串列表。

"""

# 1. 统计字符串中 n 的出现次数:
strtemp = 'ab2b3n5n2n67mm4n2'
print(len(strtemp.split('n')) - 1)  # 4

# 2. 以下演示以 + 和 _ 符号为分隔符：
str = "Chris_iven+Chris_jack+Chris_lusy"
print(str.split("+"))  # ['Chris_iven', 'Chris_jack', 'Chris_lusy']
print(str.split("_"))  # ['Chris', 'iven+Chris', 'jack+Chris', 'lusy']

# 3. 利用re模块分割含有多种分割符的字符串：
import re

a = 'Beautiful, is; better*than\nugly'
# 四个分隔符为：,  ;  *  \n
x = re.split(',|; |\*|\n', a)  # |是隔开符号
print(x)  # ['Beautiful', ' is', 'better', 'than', 'ugly']
