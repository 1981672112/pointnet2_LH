import torch
import torch.nn as nn

a = torch.arange(12, dtype=torch.float).reshape(2, 2, 3)
b = torch.ones(12).reshape(2, 2, 3)
b1 = torch.ones(12).reshape(2, 3, 2)  # .int()
c = torch.zeros(12).reshape(2, 2, 3)
print(a)
print(b)
print(c)
# out = torch.bmm(a, b)  # RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [2, 3] but got: [2, 2].


out = torch.bmm(a, b1)
print(out)

"""
v = torch.tensor([0])
m = nn.Linear(1, 10)
m(v)
注意到导致报错的代码: output = input.matmul(weight.t())
因为input也就是我们的v是torch.long类型的而weight是torch.float类型
所以在做矩阵乘法的时候这两种类型的不一致导致了报错

解决方法就是把v的dtype显示地设置成torch.float代码就成功运行了：

# dtype=torch.float必不可少
v = torch.tensor([0], dtype=torch.float)
m = nn.Linear(1, 10)
m(v)
Out[11]: 
tensor([-0.0628, -0.2544,  0.1313, -0.9293, -0.1259, -0.3151,  0.0729, -0.3097,
         0.8988,  0.1230], grad_fn=<AddBackward0>)
1
2
3
4
5
6
7
8
9
"""
v = torch.tensor([0], dtype=torch.float)
m = nn.Linear(1, 10)
o = m(v)
print(o)
