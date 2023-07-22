import torch

batch_size = 2
num_points = 1024
x = torch.randn(2 * 3 * 1024).reshape(2, 3, 1024)
device = torch.device('cuda')
# t1 = torch.arange(0, batch_size, device=device)
# t2 = torch.arange(0, batch_size, device=device).view(-1, 1, 1)
idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
# print(t1, type(t1), t1.shape)
# print(t2, type(t2), t2.shape)

print(idx_base, type(idx_base), idx_base.shape)
b, c, k, n = 2, 3, 30, 1024
idx = torch.arange(b * k * n).reshape(b, n, k)
print(idx, 'idx')
idx = idx.to(device) + idx_base
# idx1 = idx.to(device) + idx_base
idx2 = idx
print(idx2)

idx = idx.view(-1)  # 61440

_, num_dims, _ = x.size()  # num_dims=3

x1 = x.transpose(2, 1).contiguous()  # x=2 3 1024 变成 2 1024 3

tn = x1.view(batch_size * num_points, -1)
neighbor = x1.view(batch_size * num_points, -1)[idx, :]
print('1')
