"""
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
    -> (Tensor, LongTensor)

input：一个tensor数据
k：指明是得到前k个数据以及其index
dim： 指定在哪个维度上排序， 默认是最后一个维度
largest：如果为True，按照大到小排序； 如果为False，按照小到大排序
sorted：返回的结果按照顺序返回
out：可缺省，不要

"""
import torch

pred = torch.randn((4, 5))
print(pred)
values, indices = pred.topk(1, dim=1, largest=True, sorted=True)
print(indices)

# 用max得到的结果，设置keepdim为True，避免降维。因为topk函数返回的index不降维，shape和输入一致。
_, indices_max = pred.max(dim=1, keepdim=True)

print(indices_max == indices)
# pred
# tensor([[-0.1480, -0.9819, -0.3364, 0.7912, -0.3263],
#         [-0.8013, -0.9083, 0.7973, 0.1458, -0.9156],
#         [-0.2334, -0.0142, -0.5493, 0.0673, 0.8185],
#         [-0.4075, -0.1097, 0.8193, -0.2352, -0.9273]])

# indices, shape为 【4,1】,
# tensor([[3],  # 【0,0】代表 第一个样本最可能属于第一类别
#         [2],  # 【1, 0】代表第二个样本最可能属于第二类别
#         [4],
#         [2]])

# indices_max等于indices
# tensor([[True],
#         [True],
#         [True],
#         [True]])
