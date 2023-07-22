import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)  # 采样点xyz
    new_xyz_points = index_points(points, fps_idx)  # 采样点的特征

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]邻居点xyz
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # 相对位置

    if points is not None:
        grouped_points = index_points(points, idx)  # 邻居点的特征
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.lastconv = nn.Conv2d(mlp[-1] + 3, mlp[-1], 1)
        self.bn = nn.BatchNorm2d(mlp[-1])

        # if nsample is not None:
        #     self.pa = Point_SA(nsample)
        # else:
        #     self.pa = None

        self.pa = Point_SAC(mlp[-1])

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:  # False
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)# 采样xyz 邻居点特征，相对位置
            new_xyz, new_points, relative_xyz = group_feature(self.npoint, self.radius, self.nsample, xyz, points)  # 采样xyz 邻居点特征，相对位置
            new_relative = sample_and_group_relative(self.npoint, self.radius, self.nsample, xyz, points)  # 邻居和采样相对特征
            new_points = torch.cat([new_points, new_relative], dim=-1)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D 6+6, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        if self.group_all is False:
            new_points = torch.cat([new_points, relative_xyz.permute(0, 3, 2, 1)], dim=-3)
            new_points = F.relu(self.bn(self.lastconv(new_points)))

        # if self.pa is not None:
        #     new_points = self.pa(new_points)
        # else:
        #     pass

        new_points = torch.max(new_points, 2)[0]  # 2 128 512 bcn
        new_points = self.pa(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


def sample_and_group_relative(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)  # 采样点xyz
    new_xyz_points = index_points(points, fps_idx)  # 采样点的特征

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_points = index_points(points, idx)  # 邻居点的特征

    C1 = new_xyz_points.shape[-1]
    new_xyz_points = new_xyz_points.view(B, S, 1, C1)
    new_xyz_points = new_xyz_points.repeat(1, 1, nsample, 1)

    relative_feature = grouped_points - new_xyz_points
    # relative_feature = torch.cat([grouped_points_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    return relative_feature


def group_feature(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)  # 采样点xyz

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]邻居点xyz

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # 相对位置

    if points is not None:
        grouped_points = index_points(points, idx)  # 邻居点的特征
        # new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]

    if returnfps:
        return new_xyz  # , new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, grouped_points, grouped_xyz_norm


class Point_SA(nn.Module):
    def __init__(self, channels):
        super(Point_SA, self).__init__()
        self.v_conv = nn.Conv1d(channels, channels, 1)  # 由Linear来
        # self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)
        self.softmax = nn.Softmax(dim=-2)
        self.affine_alpha = nn.Parameter(torch.ones([1, channels, 1]))
        self.affine_beta = nn.Parameter(torch.zeros([1, channels, 1]))

    def forward(self, x):  # x [2,128,2048]
        B, C, N = x.shape
        x = self.act(self.after_norm(self.v_conv(x)))  # 通道注意
        attention = self.softmax(x)
        attention1 = attention / (1e-9 + attention.sum(dim=-1, keepdims=True))
        x = x * attention1
        x = x + self.affine_alpha * x + self.affine_beta
        return x


class Point_SAC(nn.Module):
    def __init__(self, channels):
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

class Point_SACgai(nn.Module):
    def __init__(self, channels):
        super(Point_SACgai, self).__init__()
        self.v_conv = nn.Conv1d(channels, channels, 1)  # 由Linear来
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-2)
        self.affine_alpha = nn.Parameter(torch.ones([1, channels, 1]))
        self.affine_beta = nn.Parameter(torch.zeros([1, channels, 1]))

    def forward(self, x,p):  # x [2,128,2048]
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

class Point_SaA(nn.Module):
    def __init__(self, channels):
        super(Point_SaA, self).__init__()
        self.v_conv = nn.Conv1d(channels, channels, 1)  # 由Linear来
        # self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)
        self.softmax = nn.Softmax(dim=-2)
        self.affine_alpha = nn.Parameter(torch.ones([1, channels, 1]))
        self.affine_beta = nn.Parameter(torch.zeros([1, channels, 1]))

    def forward(self, x):  # x [2,128,2048]
        B, C, ns, N = x.shape
        x = x.reshape(B * C, ns, N)
        x = self.act(self.after_norm(self.v_conv(x)))  # 对点卷积 不是通道
        attention = self.softmax(x)
        attention1 = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x = x * attention1
        # x = self.affine_alpha * x + self.affine_beta
        x = x + self.affine_alpha * x + self.affine_beta
        x = x.reshape(B, C, ns, N)
        return x


if __name__ == '__main__':
    x = torch.randn((2, 8, 2, 32))
    pa = Point_SA(2)
    # x=x.reshape(16,2,32)
    x = pa(x)


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # 128 32 由Linear来
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # 128 32 由Linear来
        self.q_conv.weight = self.k_conv.weight

        self.v_conv = nn.Conv1d(channels, channels, 1)  # 128 128 由Linear来
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x [2,128,2048]
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c [2,128,2048]>[2,32,2048]>[2,2048,32]
        x_k = self.k_conv(x)  # b, c, n   [2,32,2048]
        x_v = self.v_conv(x)  # b,c,n [2,128,2048]
        energy = torch.bmm(x_q, x_k)  # b, n, n  [2,2048,2048]
        attention = self.softmax(energy)  # [2,2048,2048] 权重归一化
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # AttentionMap
        x_r = torch.bmm(x_v, attention)  # b, c, n[2,128,2048]AttentionFeature
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # (x - x_r)为实线 (x_r)则为虚线
        x = x + x_r  # output
        return x
