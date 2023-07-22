import torch
import numpy as np

'''1 一种运算有多种语法。在下面的示例中，我们将研究加法运算'''
# 1加法：形式一
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print('x', x, '\n', 'y', y)
print(x + y)
# 2加法：形式二
print(torch.add(x, y))

# 3加法：给定一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 加法：原位/原地操作（in-place）
# adds x to y
# 注意：任何一个就地改变张量的操作后面都固定一个 _ 。例如 x.copy_（y）， x.t_（）将更改x
y.add_(x)
print(y)

'''2 也可以使用像标准的 NumPy 一样的各种索引操作：'''
print(x[:, 1])
# 改变形状：如果想改变形状，可以使用 torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
# 如果是仅包含一个元素的 tensor，可以使用 .item（） 来得到对应的 python 数值
x = torch.randn(1)
print(x)
print(x.item())

'''3 将一个 Torch 张量转换为一个 NumPy 数组是轻而易举的事情，反之亦然。
Torch 张量和 NumPy数组将共享它们的底层内存位置，因此当一个改变时，另外也会改变。'''
# （1）将 torch 的 Tensor 转换为 NumPy 数组
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# NumPy细分改变里面的值的,当a改变时，b也随之改变
a.add_(1)
print(a)
print(b)
# （2）将 NumPy 数组转化为Torch张量
# 改变 NumPy 分配自动改变 Torch 张量
# 注：CPU上的所有张量（ CharTensor 除外）都支持与 Numpy 的相互转换

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
# （3）CUDA上的张量
# 张量可以使用 .to方法移动到任何设备（device）上
x = torch.randn(4, 4)
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)  # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # `.to`也能在移动时改变dtype

'''4 桥接NumPy '''
# 将一个 Torch 张量转换为一个 NumPy 数组是轻而易举的事情，反之亦然。
# Torch 张量和 NumPy数组将共享它们的底层内存位置，因此当一个改变时，另外也会改变。
# （1）将 torch 的 Tensor 转换为 NumPy 数组
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# NumPy细分改变里面的值的,当a改变时，b也随之改变
a.add_(1)
print(a)
print(b)
