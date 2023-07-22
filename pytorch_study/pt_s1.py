# from __future__ import print_function

import torch

"""
    torch.rand和torch.randn有什么区别？ 
    一个均匀分布，一个是标准正态分布。

"""

'''1 创建一个没有初始化的 5 * 3 矩阵'''
x = torch.empty(5, 3)
print(x)

'''2 创建一个随机初始化矩阵'''
x = torch.rand(5, 3)
print(x)

'''3 构造一个填满 0 且数据类型为 long 的矩阵： '''
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

'''4 直接从数据构造张量： 
或根据现有的 tensor 建立新的 tensor 。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如 dtype 等：'''
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)  # 重载 dtype!
print(x)  # 结果size一致 注意：torch.Size 本质上还是 tuple ，所以支持 tuple 的一切操作

'''5 获取张量的形状：'''
print(x.size())

# randint(low=0, high, size, out=None, dtype=None)
# randint_like(input, low=0, high, dtype=None)
# 整数范围[low, high)
t1 = torch.randint(1, 4, (2, 3, 2))  # 形状写成[2,3,2]也行
t2 = torch.randint_like(t1, 4)
"""
torch.randint_like 返回具有与Tensor input 相同形状的张量，其中填充了均匀地在 low （包含）和 high （不含）之间生成的随机整数。
"""
print(t1)
print(t2)
