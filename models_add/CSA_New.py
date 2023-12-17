# -*- coding: utf-8 -*-


import torch
# from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
#     NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

import torch.nn as nn

from torch.nn import functional as F


class CSA(nn.Module):
    """ Position attention module"""

    def __init__(self, qk_channel, vx_channel, blinear=False):
        super(CSA, self).__init__()
        self.blinear = blinear  # 插值参数

        self.query_conv = nn.Conv1d(in_channels=qk_channel, out_channels=qk_channel, kernel_size=1, bias=False)
        self.key_conv = nn.Conv1d(in_channels=qk_channel, out_channels=qk_channel, kernel_size=1, bias=False)
        self.value_conv = nn.Conv1d(in_channels=vx_channel, out_channels=vx_channel, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_h):
        """
            inputs :
                x : 低维特征  h高维特征
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        b, c, n = x.size()
        m_batchsize, C, N = x_h.size()

        proj_query = self.query_conv(x).view(b, -1, n).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, n)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        if self.blinear == False:  # 高语义HW不匹配，要向上插值，但是HW小于hw
            proj_value = self.value_conv(x_h).view(m_batchsize, -1, N)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, N)
            out = self.gamma * out + x_h
            return out
        else:
            # x_h = F.interpolate(x_h, size=(h, w), mode='bilinear', align_corners=True)
            proj_value = self.value_conv(x_h).view(m_batchsize, -1, n)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, n)
            out = self.gamma * out + x_h
            return out


if __name__ == '__main__':
    inputs_L = torch.randn(2, 64, 1024)
    inputs_H = torch.randn(2, 64, 1024)
    net = CSA(64, 64)
    o1 = net(inputs_L, inputs_H)
    print(o1.size())
    print(o1[0].size())
    print(o1[1].size())
    print(o1[:, 0:10, :].size())
