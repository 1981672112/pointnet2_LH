"""
假设我目前的张量的shape为[ B , N , C ] = [ 32 , 512 , 64 ]

N表示点的数量，我要从中随机选取S个点，且为不重复采样。

"""

import torch
import torch.nn as nn
import random

# BNC
a = torch.randn((2, 512, 64))
print(a.shape)  # torch.Size([32, 512, 64])
B, N, C = a.shape
S = 64
index = torch.LongTensor(random.sample(range(N), S))
print(index)
b = torch.index_select(a, 1, index)
print(b.shape)  # torch.Size([32, 64, 64])
