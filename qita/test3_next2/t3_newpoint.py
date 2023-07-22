import torch
import torch.nn as nn

import random


class Layer_Reduce(nn.Module):
    def __init__(self, channels):
        super(Layer_Reduce, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_2c = nn.Conv1d(channels // 2, channels // 4, 1, bias=False)  # ###
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.softmax1 = nn.Softmax(dim=-2)  # ####

        self.alpha = nn.Parameter(torch.ones([1, channels, 1]))
        self.beta = nn.Parameter(torch.zeros([1, channels, 1]))

    def forward(self, x):
        # b, n, c
        B, C, N = x.shape
        x_q = self.q_conv(x).permute(0, 2, 1)

        # b, c, n
        S = C // 4
        index = torch.LongTensor(random.sample(range(C), S))
        x1 = torch.index_select(x, 1, index)
        x_k1 = x1

        x_k = self.k_conv(x)
        x_k = torch.cat((x_k, x_k1), dim=1)  # b 2c n
        x_k = self.k_2c(x_k)
        x_k = self.softmax1(x_k)

        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        # b, c, n
        x_r = torch.bmm(x_v, attention)

        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r
        return x


if __name__ == '__main__':
    input = torch.randn((2, 64, 1024))
    p = Layer_Reduce(64)
    out = p(input)
    print(out.shape)
