import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")


# %matplotlib inline
# “%matplot inline” 是为了在 Jupyter 中正常显示图形，若没有这行代码，图形显示不出来的。


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


x = torch.Tensor(
    [
        [1, 2, 3],  # q1, k1, v1
        [3, 2, 1],  # q2, k2, v2
        [3, 2, 1],  # q3, k3, v3
        [3, 2, 1],  # q4, k4, v4
    ]
)

print("=" * 15, "w/o mask", "=" * 15)
_, p_attn = attention(x, x, x, mask=None)
print("p_attn:\n", p_attn)
plt.figure(figsize=(6, 6))
seaborn.heatmap(p_attn.numpy(), annot=True, cmap="YlOrBr", square=True)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.show()

print("=" * 15, " wz mask", "=" * 15)
mask = subsequent_mask(x.size(0))
_, p_attn = attention(x, x, x, mask)
print("p_attn:\n", p_attn)
plt.figure(figsize=(6, 6))
seaborn.heatmap(p_attn[0].numpy(), annot=True, cmap="YlOrBr", square=True)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.show()
