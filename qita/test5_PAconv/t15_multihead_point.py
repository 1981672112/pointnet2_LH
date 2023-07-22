import torch
import torch.nn as nn
import numpy as np
import random
from t10_ScaleDotProductAttention import ScaledDotProductAttention

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Layer_Reduce(nn.Module):
    def __init__(self, channels):
        super(Layer_Reduce, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)

        self.k_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.k_2c = nn.Conv1d(channels // 4, channels // 8, 1, bias=False)  # ###

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.softmax1 = nn.Softmax(dim=-2)  # ####

        self.alpha = nn.Parameter(torch.ones([1, channels, 1]))
        self.beta = nn.Parameter(torch.zeros([1, channels, 1]))

    def forward(self, q, k, v):
        # b, n, c
        B, C, N = k.shape
        x_q = self.q_conv(q).permute(0, 2, 1)

        # b, c, n
        S = C // 8
        index = torch.LongTensor(random.sample(range(C), S))  # .to(device)  # train 无 # run有#
        x1 = torch.index_select(k, 1, index)

        x_k1 = x1
        x_k = self.k_conv(k)
        x_k = torch.cat((x_k, x_k1), dim=1)  # b 2c n cat(0.125c+0.125c)>0.25c
        x_k = self.k_2c(x_k)  # 0.25c>0.125c
        x_k = self.softmax1(x_k)

        x_v = self.v_conv(v)

        # b, n, n
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)

        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        # x = x + self.alpha * x_r + self.beta
        x = self.alpha * x_r + self.beta
        return attention, x


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


class MultiHead_LayerReduce(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, channels, d_v_, d_k, d_v, d_o):
        super(MultiHead_LayerReduce, self).__init__()
        'n_head, d_k_, d_v_, d_k, d_v, d_o = '
        '8       128   64    256  128  128'
        # dk=channels
        self.n_head = n_head  # 8
        self.d_k = channels  # 256
        # self.d_v = int(0.5 * channels)  # 128
        self.d_v = channels  # 128

        'qkv'
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.fc_q = nn.Linear(channels, n_head * self.d_k)  # 128>8*256(2048)
        self.fc_k = nn.Linear(channels, n_head * self.d_k)  # 128>8*256(2048)
        self.fc_v = nn.Linear(channels, n_head * self.d_v)  # 64>8*128(1024)
        # self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
        self.attention = Layer_Reduce(channels)
        self.fc_o = nn.Linear(n_head * self.d_v, d_o)  # 8*128 128

    # def forward(self, q, k, v, mask=None):
    def forward(self, x, mask=None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v
        '头=8，d_q, d_k, d_v=256 256 128'
        'q 2 2 128'
        'k 2 4 128'
        'v 2 2 64'

        'x = bcn '
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        print('0', id(q))

        'bcn 2 64 1024'
        batch, d_q_, n_q = q.size()  # 2 2 128
        batch, d_k_, n_k = k.size()  # 2 4 128
        batch, d_v_, n_v = v.size()  # 2 4 64

        'b c n > b n c*h'
        q = self.fc_q(q.transpose(2, 1))  # 1.单头变多头 每个张量经过线性层 # 2 2 128》# 2 2 256*8 d_q_>n_head * d_k
        k = self.fc_k(k.transpose(2, 1))  # 2 4 128>2 4 8*256
        v = self.fc_v(v.transpose(2, 1))  # 2 4 64>2 4 8*128
        print('1', id(q))

        'b n c*h>b*h c n'
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, d_q, n_q)  # >>nh b nq dq >> nh*b nq dq
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, d_k, n_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, d_v, n_v)
        print('2', id(q))
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        attn, output = self.attention(q, k, v)  # 2.当成单头注意力求输出
        print('1', id(output))
        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)  # 3.Concat
        print('2', id(output))
        output = self.fc_o(output).transpose(2, 1)  # 4.仿射变换得到最终输出
        print('3', id(output))
        # attn, output=16 1024 1024 16 1024 32 与同v维度相同
        return attn, output


if __name__ == '__main__':
    '''
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
    '''
    input = torch.randn((2, 64, 1024))
    # p = Layer_Reduce(64)
    p = MultiHead_LayerReduce(8, 64, 64, 64, 64, 64)
    att, out = p(input)
    print(out.shape)

    # i = torch.randn((2, 64 + 3, 1024, 30))
    # print(i.shape[0])
    # pp = PositionScore(i.shape[0], i.shape[1], i.shape[2], i.shape[3], i.shape[1])
    # ou = pp(i)
    # print(ou.shape)
