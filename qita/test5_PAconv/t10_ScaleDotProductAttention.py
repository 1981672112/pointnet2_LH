import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))  # 1.Matmul 2 2 128 2 128 4 》= 2 2 4
        u = u / self.scale  # 2.Scale
        'masked_fill方法有两个参数，mask和value，'
        'mask是一个pytorch张量（Tensor），元素是布尔值，value是要填充的值，'
        '填充规则是mask中取值为True位置对应于self的相应位置用value填充。'
        print(u)
        if mask is not None:
            u = u.masked_fill(mask, -np.inf)  # 3.Mask # np.inf 表示+∞
        print(u)

        attn = self.softmax(u)  # 4.Softmax
        output = torch.bmm(attn, v)  # 5.Output
        return attn, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64

    batch = 2
    q = torch.randn(batch, n_q, d_q)  # 2 2 128
    k = torch.randn(batch, n_k, d_k)  # 2 4 128
    v = torch.randn(batch, n_v, d_v)  # 2 4 64

    mask = torch.zeros(batch, n_q, n_k).bool()  # 2 2 4全False
    print('hhhhhh', np.power(d_k, 0.5), 'hhhhhhhhh')  # 11.31
    attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v, mask=mask)

    print(attn)
    print(output)
