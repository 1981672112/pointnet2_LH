import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

# 会比较shape 和 值是否都相同，相同则只返回一个TRUE或者FALSE
print(a.shape, b.shape)
print('\n', torch.equal(a, b))  # False

# 以下的是要对每一个元素进行比较，
print('\n', torch.eq(a, b))
"""
 tensor([[False, False, False],
        [False, False, False]])

"""

print('大于等于\n', torch.ge(a, b))  # a大于等于b
print('大于\n', torch.gt(a, b))
print('小于等于\n', torch.le(a, b))
print('小于\n', torch.lt(a, b))
print('不等于\n', torch.ne(a, b))
