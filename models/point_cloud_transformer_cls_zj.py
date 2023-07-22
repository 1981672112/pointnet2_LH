import torch
import torch.nn as nn
from pointnet2_utils import farthest_point_sample, index_points, square_distance
import torch.nn.functional as F


class point_could_transformer_cfg:
    num_class = 40
    input_dim = 6


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()

        assert num_class == 40, '仅支持modelnet40'
        assert normal_channel == True, '必须使用RGB信息'
        cfg = point_could_transformer_cfg
        self.net = Point_Cloud_TransformerCls(cfg)
        "可能这里需要一下初始化"

    def forward(self, x):
        x = self.net(x)
        x = F.log_softmax(x, -1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # print(pred.shape)
        # print(target.shape)
        total_loss = F.nll_loss(pred, target)

        return total_loss


class Point_Cloud_TransformerCls(nn.Module):
    def __init__(self, cfg):
        super(Point_Cloud_TransformerCls, self).__init__()
        output_channels = cfg.num_class  # 10
        d_points = cfg.input_dim  # 6
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)  # 升维 飞机后的2个LBR 6>64 后面2个SG
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention_gai()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280 + 256, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # print(x.size())
        x = x.permute(0, 2, 1)  # # x=[2,1024,6]
        xyz = x[..., :3]  # # xyz=[2,1024,3]
        x = x.permute(0, 2, 1)  # # x=[2,6,1024]
        batch_size, _, _ = x.size()  # 2,3,1024
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N  # 2,64,1024 # 卷积进行特征升维  对应论文中的图4 的LBR
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N # 2,64,1024

        x = x.permute(0, 2, 1)  # 2,1024,64 x=BNC在采样
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=8, xyz=xyz,
                                                points=x)
        # 对应论文图4中的SG  2,512,3     2,512,32，64+64        使用两个级联的 SG 层来逐渐扩大特征聚合过程中感受野
        feature_0 = self.gather_local_0(new_feature)  # 2,128，512 BCN #512个点 每个点128个特征
        "sample_and_group + self.gather_local_0 就是PointNet++里面的一个setabstraction层（将坐标和特征的拼接，变成了单独对特征的拼接） "

        feature = feature_0.permute(0, 2, 1)  # 变化后feature = 2,512,128
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=8, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)  # 2,256,256 BCN 点数减半  特征翻倍

        x = self.pt_last(feature_1)  # stack sa 首先对输入特征进行两次变换，然后进行四次SA  最后将他们的结果拼接起来  2,1280,256
        x = torch.cat([x, feature_1], dim=1)  # 在embedding之后的feature进行拼接
        x = self.conv_fuse(x)  # 2,1280,256 到2 1024 256
        x = torch.max(x, 2)[0]  # 2,1024那个点最能代表这个维度特征
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x


def sample_and_group(npoint, nsample, xyz, points):
    # points=x 2,1024,64  xyz 2,1024,3
    B, N, C = xyz.shape  # 2,1024,3
    S = npoint  # 512

    fps_idx = farthest_point_sample(xyz, npoint)  # 返回[B, npoint] # 2,512
    "根据坐标利用FPS算法找到质心点的索引 ，然后去原始的xyz 和points 去索引对应位置的点和特征"
    new_xyz = index_points(xyz, fps_idx)  # 2,512,3  坐标的中心点
    new_points = index_points(points, fps_idx)  # 2,512,64  特征的中心点

    dists = square_distance(new_xyz, xyz)  # B x npoint x N  2,512,1024   512个中心点与全部点的距离
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K  2,512,32  #取它前(一行（1024）最小的32个数的索引取出来)三十二个邻居的索引

    grouped_points = index_points(points, idx)  # 2,512,32,64 (前一个) 找到特征对应的邻域
    grouped_points_norm = grouped_points - new_points.view(B, S, 1,
                                                           -1)
    # 2,512,1,64  #  （后一个）2,512,32,64 将特征领域和特征中心点相减 即 领域中的每个中心点都与领域的特征点相减  转换为相对特征点

    # new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)],dim=-1)  # 2,512,32,64+64 将相对特征点与中心点进行拼接
    new_points = torch.cat([grouped_points_norm, grouped_points], dim=-1)  # 2,512,32,64+64 将相对特征点与中心点进行拼接

    return new_xyz, new_points  # 2,512,3     2,512,32，64+64  返回新的坐标和拼接后的点(的特征)


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入是之前拼接的特征点
        b, n, s, d = x.size()  # 2,512,32,128
        x = x.permute(0, 1, 3, 2)  # 2,512,128,32
        x = x.reshape(-1, d, s)  # 1024,128,32 拉伸送入卷积层进行
        batch_size, _, N = x.size()  # N32 b_s 1024
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N1024，128，32
        x = torch.max(x, 2)[0]  # 1024，128 取32个邻居特征中 每个维度最大的一个值  如 第1行就是  32个邻居中 第1个位置最大的值 [1024,128]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)  # [1024,128][2,512,128][2,128,512]
        return x  # 2,128,512 变换回去


# class SA_Layer(nn.Module):
#     def __init__(self, channels):
#         super(SA_Layer, self).__init__()
#         self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
#         self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
#         self.q_conv.weight = self.k_conv.weight
#
#         self.v_conv = nn.Conv1d(channels, channels, 1)
#         self.trans_conv = nn.Conv1d(channels, channels, 1)
#         self.after_norm = nn.BatchNorm1d(channels)
#         self.act = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x=(2,256,256)2,256,256 BCN
#         x_q = self.q_conv(x).permute(0, 2, 1)  # 2,256,64 # b, n, c
#         x_k = self.k_conv(x)  # b, c, n 2,64,256
#         x_v = self.v_conv(x)  # 2,256,256
#         energy = x_q @ x_k  # b, n, n 2,256,256
#         attention = self.softmax(energy)
#         attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
#         x_r = x_v @ attention  # b, c, n
#         # print(x-x_r)
#         x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
#         x = x + x_r
#         return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super(StackedAttention, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layerzj(channels, 256)
        self.sa2 = SA_Layerzj(channels, 256)
        self.sa3 = SA_Layerzj(channels, 256)
        self.sa4 = SA_Layerzj(channels, 256)
        self.sa5 = SA_Layerzj(channels, 256)

        self.relu = nn.ReLU()

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()
        # x.size()  2,256,256
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        # 将输入特征进行两次变换  ---->2,256,256   接下来连做四次SA
        x_save = x
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = self.sa5(x4)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        return x


class SA_Layerzj(nn.Module):
    def __init__(self, channels, points):
        super(SA_Layerzj, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # 128 32 由Linear来
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # 128 32 由Linear来
        self.q_conv.weight = self.k_conv.weight

        self.v_conv = nn.Conv1d(channels, channels, 1)  # 128 128 由Linear来
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # self.affine_alpha = nn.Parameter(torch.ones([1, 1, points]))  # 64 (1,1,1,64) 都为1
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, points]))  # 64 (1,1,1,64) 都为0

    def forward(self, x):  # x [2,128,2048]
        _, _, N = x.shape  # N是最远点采样数量
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c [2,128,2048]>[2,32,2048]>[2,2048,32]
        x_k = self.k_conv(x)  # b, c, n   [2,32,2048]

        x_v = self.v_conv(x)  # b,c,n [2,128,2048]

        x_w = torch.max(x_v, dim=2, )[0]  # .unsqueeze(dim=-1).repeat(1,1,N)
        x_w = self.softmax(x_w).unsqueeze(dim=-1).repeat(1, 1, N)

        energy = torch.bmm(x_q, x_k)  # b, n, n  [2,2048,2048]
        attention = self.softmax(energy)  # [2,2048,2048] 权重归一化
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # AttentionMap

        # attention=attention*x_w+self.affine_beta # 错 前面维度不匹配

        x_r = torch.bmm(x_v, attention)  # b, c, n[2,128,2048]AttentionFeature
        x_r = x_r * x_w + self.affine_beta  # ++
        t = torch.mean(x_r, dim=2).unsqueeze(dim=-1).repeat(1, 1, N)
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # (x - x_r)为实线 (x_r)则为虚线
        t1 = self.act(self.after_norm(self.trans_conv(x - t)))
        # x_r = self.affine_alpha * x_r + self.affine_beta
        # t1 = self.affine_alpha * t1 + self.affine_beta
        # x = x + x_r + t1  # output
        x = x + t1  # output
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
        # x_max = torch.max(x, 1)[0].repeat(1,C,1).view(B,C,N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x2)))
        x = x + self.affine_alpha * x_r + self.affine_beta
        return x


class StackedAttention_gai(nn.Module):
    def __init__(self, channels=256):
        super(StackedAttention_gai, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = Point_SAC(channels, 256)
        self.sa2 = Point_SAC(channels, 256)
        self.sa3 = Point_SAC(channels, 256)
        self.sa4 = Point_SAC(channels, 256)
        self.sa5 = Point_SAC(channels, 256)

        self.relu = nn.ReLU()

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()
        # x.size()  2,256,256
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        # 将输入特征进行两次变换  ---->2,256,256   接下来连做四次SA
        x_save = x
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = self.sa5(x4)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        return x


if __name__ == '__main__':
    net = get_model(num_class=40)
    # print(net)
    i = torch.randn((2, 6, 1024))
    o = net(i)
    print(o.size())
