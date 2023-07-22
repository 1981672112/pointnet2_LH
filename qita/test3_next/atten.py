import torch
import torch.nn as nn
import torch.nn.functional as F


class CSA_Layer(nn.Module):
    def __init__(self, channels, activate_function):
        super(CSA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        # self.trans_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = activate_function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, position):
        x_q = self.q_conv(position).permute(0, 2, 1)[:, :, :, None]  # b, n, c,1
        # x_k = self.k_conv(position).permute(0, 2, 1)[:, :, None, :]  # b, n, 1,c
        x_k = self.k_conv(position).permute(0, 2, 1)[:, :, None, :]  # b, n, 1,c
        x_v = self.v_conv(feature)

        energy = torch.matmul(x_q, x_k)  # b,n c c
        energy = torch.sum(energy, dim=-2, keepdim=False)  # b n c
        energy = energy / (1e-9 + energy.sum(dim=-1, keepdim=True))
        attention = self.softmax(energy.permute(0, 2, 1))  # bcn

        x_r = torch.mul(attention, x_v)  # b, c, n
        x = (x_r + feature)
        # x = self.act(self.after_norm(self.trans_conv(feature + x_r)))

        return x


class Layer(nn.Module):
    def __init__(self, channels, activate_function):
        super(Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        # self.trans_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = activate_function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, position):
        x_q = self.q_conv(position).permute(0, 2, 1)[:, :, :, None]  # b, n, c,1
        # x_k = self.k_conv(position).permute(0, 2, 1)[:, :, None, :]  # b, n, 1,c
        x_k = position.permute(0, 2, 1)[:, :, None, :]  # b, n, 1,c
        x_v = self.v_conv(feature)

        energy = torch.matmul(x_q, x_k)  # b,n c c

        energy = torch.sum(energy, dim=-2, keepdim=False)  # b n c
        energy = energy / (1e-9 + energy.sum(dim=-1, keepdim=True))
        attention = self.softmax(energy.permute(0, 2, 1))  # bcn

        x_r = torch.mul(attention, x_v)  # b, c, n
        x = (x_r + feature)
        # x = self.act(self.after_norm(self.trans_conv(feature + x_r)))

        return x


class Point_SAC(nn.Module):
    def __init__(self, channels, points):
        super(Point_SAC, self).__init__()
        self.v_conv = nn.Conv1d(channels, channels, 1)  # 由Linear来
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-2)
        self.affine_alpha = nn.Parameter(torch.ones([1, channels, 1]))
        self.affine_beta = nn.Parameter(torch.zeros([1, channels, 1]))

    def forward(self, x):  # x [2,128,2048]
        B, C, N = x.shape
        x0 = self.v_conv(x)  # 通道注意
        x1 = x.permute(0, 2, 1)

        energy = torch.bmm(x1, x0)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x2 = torch.bmm(x0, attention)

        x_r = self.act(self.after_norm(self.trans_conv(x - x2)))
        x = x + self.affine_alpha * x_r + self.affine_beta
        return x


if __name__ == '__main__':
    # inputs = torch.randn((2, 6, 2048))
    # o = get_model(50).eval()
    # # print(o)
    # print(o(inputs).size())
    # list_ = [0.3, 0.7, -1, 2, 1.2]
    # tensor_list = torch.as_tensor(list_)
    # softmax = nn.Softmax(dim=-1)
    # print(softmax(tensor_list))

    inputs = torch.randn((2, 6, 2048))
    position = torch.randn((2, 6, 2048))
    # o = get_model(50).eval()
    # print(o)
    # print(o(inputs).size())
    # l = Point_SAC(6, 'relu')
    l = Layer(6, 'relu')

    out = l(inputs,position)
    print(out, out.shape)

    # list_ = [0.3, 0.7, -1, 2, 1.2]
    # tensor_list = torch.as_tensor(list_)
    # softmax = nn.Softmax(dim=-1)
    # print(softmax(tensor_list))
