import torch
import torch.nn as nn
import numpy as np
from t10_ScaleDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()
        'n_head, d_k_, d_v_, d_k, d_v, d_o = '
        '8       128   64    256  128  128'
        self.n_head = n_head  # 8
        self.d_k = d_k  # 256
        self.d_v = d_v  # 128

        self.fc_q = nn.Linear(d_k_, n_head * d_k)  # 128>8*256(2048)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)  # 128>8*256(2048)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)  # 64>8*128(1024)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)  # 8*128 128

    def forward(self, q, k, v, mask=None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v
        '头=8，d_q, d_k, d_v=256 256 128'

        batch, n_q, d_q_ = q.size()  # 2 2 128
        batch, n_k, d_k_ = k.size()  # 2 4 128
        batch, n_v, d_v_ = v.size()  # 2 4 64

        q = self.fc_q(q)  # 1.单头变多头 每个张量经过线性层 # 2 2 128》# 2 2 256*8 d_q_>n_head * d_k
        k = self.fc_k(k)  # 2 4 128>2 4 8*256
        v = self.fc_v(v)  # 2 4 64>2 4 8*128
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)  # >>nh b nq dq >> nh*b nq dq
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)  # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)  # 3.Concat
        output = self.fc_o(output)  # 4.仿射变换得到最终输出

        return attn, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 128, 128, 64
    batch = 2

    q = torch.randn(batch, n_q, d_q_)  # 2 2 128
    k = torch.randn(batch, n_k, d_k_)  # 2 4 128
    v = torch.randn(batch, n_v, d_v_)  # 2 4 64
    mask = torch.zeros(batch, n_q, n_k).bool()  # 2 2 4 all False

    mha = MultiHeadAttention(n_head=8, d_k_=128, d_v_=64, d_k=256, d_v=128, d_o=128)
    attn, output = mha(q, k, v, mask=mask)

    print(attn.size())
    print(output.size())
