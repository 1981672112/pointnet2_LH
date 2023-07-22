import torch

a = torch.arange(1, 8 * 3 * 4 + 1).view(8, 3, 4)  # bcn
print(a, 'a')
a1 = a.transpose(2, 1)  # 2 4 3 矩阵转置
a2 = a.transpose(1, 0)  # 3 2 4 重新分批
a3 = a.transpose(2, 0)  # 4 3 2
print(a1, 'a1')
print(a2, 'a2')
print(a3, 'a3')

'1 equal_a尽然和a3相同'
# equal_a = a.transpose(2, 1)
# equal_a = equal_a.transpose(0, 1)
# equal_a = equal_a.transpose(2, 1)
# print(equal_a, 'equal_a')
# 234=ABC 432(CBA) 怎么来 初始：ABC 1步>ACB 2步>CAB 3步>CBA

'2 变维度'
a4 = a.view(2 * 4, -1)
print(a4, 'a4', a4.shape, a4.size())  # 重新摆放
a5 = a.view(-1, 2 * 4).contiguous()  # torch.Size([3, 8])
print(a5, 'a5', a5.shape, a5.size())  # 重新摆放

# tensor1 = torch.Tensor(1, 2, 2, 2).long()  # a5=torch.Size([3, 8]) >>torch.Size([3, 1, 2, 2, 2])
tensor1 = torch.Tensor(4, 5).long()  # a5=torch.Size([3, 8]) >>torch.Size([3, 1, 2, 2, 2])
tensor2 = torch.Tensor([0, 1, 7, 7, 1, 2, 3, 2, 1, 0]).long()  # >>torch.Size([3, 0]) 带小括号和列表()[]在单个维度里面指定
# print(tensor1, 'tensor1')
# print(tensor2, 'tensor2')

tensor = tensor2
# a6 = a5[:, tensor]
a6 = a5[tensor, :]
print(a6, 'a6', a6.size())
# tensor([   0,  649,  735,  ..., 1491, 1228, 1243], device='cuda:0')
