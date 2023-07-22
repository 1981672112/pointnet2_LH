"""
mm只能进行矩阵乘法,也就是输入的两个tensor维度只能是n xm 和m x p
bmm是两个三维张量相乘, 两个输入tensor维度是( b × n × m )和( b × m × p ) ,第一维b代表batch size，输出为(b×n×p)
matmul可以进行张量乘法, 输入可以是高维.
"""
import torch

a = torch.arange(0, 6).reshape(2, 3)
b = torch.arange(0, 6).reshape(3, 2)
print(a, '\n', b)
print(torch.mm(a, b))  # 普通2维矩阵相乘

a2 = torch.arange(0, 12).reshape(2, 2, 3)
b2 = torch.arange(0, 12).reshape(2, 3, 2)
print(a2, '\n', b2)
print(torch.bmm(a2, b2))  # 对应的批 进行普通2维矩阵相乘

a3 = torch.arange(0, 12).reshape(1, 2, 3, 2)
b3 = torch.arange(0, 12).reshape(2, 2, 3)
print(a3, '\n', b3)
print(torch.matmul(a3, b3))  # matmul可以进行张量乘法, 输入可以是高维.
