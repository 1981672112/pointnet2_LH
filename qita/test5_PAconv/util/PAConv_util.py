import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    B, _, N = x.size()  # x=2 3 1024
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # 2 1024 1024
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # 2 1 1024
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 2 1024 1024

    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

    return idx, pairwise_distance  # 返回索引和距离


def get_scorenet_input(x, idx, k):
    """(neighbor, neighbor-center)"""
    batch_size = x.size(0)  # x 2 3 1024 idx bnk =2 1024 30
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # 还是bcn 2 3 1024
    print(idx, type(idx))
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base  # 2 1024 30 + 2 (batch_size) 1 1    (0) (1024)

    idx = idx.view(-1)  # 61440 索引tensor([   0,  743,   36,  ..., 1043, 1613, 1953], device='cuda:0')

    _, num_dims, _ = x.size()  # num_dims=3

    x = x.transpose(2, 1).contiguous()  # x=2 3 1024 变成 2 1024 3

    neighbor = x.view(batch_size * num_points, -1)[idx, :]  # 2048 3>>61440 3 矩阵行列 x是值 idx是索引 从2048行中选61440行及其3列

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)  # 2 1024 30 3 #每个点的最近30邻居点特征

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # 增加第2维重复到K 好减 2 1024 30 3

    xyz = torch.cat((neighbor - x, neighbor), dim=3).permute(0, 3, 1, 2)  # b,6,n,k

    return xyz  # 返回1024(每个点)的30个邻居点xyz位置特征（3）和邻居点减去中心点的相对位置特征（3） 因为x=2 3 1024 3代表xyz特征


def feat_trans_dgcnn(point_input, kernel, m):
    """transforming features using weight matrices"""
    # following get_graph_feature in DGCNN: torch.cat((neighbor - center, neighbor), dim=3)
    B, _, N = point_input.size()  # b, 2cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1).repeat(1, 1, 2), kernel).view(B, N, m, -1)  # b,n,m,cout
    center_output = torch.matmul(point_input.permute(0, 2, 1), kernel[:point_input.size(1)]).view(B, N, m, -1)  # b,n,m,cout
    return point_output, center_output


def feat_trans_pointnet(point_input, kernel, m):
    """transforming features using weight matrices"""
    # no feature concat, following PointNet
    B, _, N = point_input.size()  # b, cin, n 特征2 64 1024
    point_output = torch.matmul(point_input.permute(0, 2, 1), kernel).view(B, N, m, -1)  # b,n,m,cout（b n cin ）(64 512)
    return point_output


class ScoreNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[16], last_bn=False):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit  # [16]
        self.last_bn = last_bn  # False
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()

        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs_nohidden = nn.Conv2d(in_channel, out_channel, 1, bias=not last_bn)
            if self.last_bn:
                self.mlp_bns_nohidden = nn.BatchNorm2d(out_channel)

        else:
            self.mlp_convs_hidden.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))  # from in_channel to first hidden
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):  # from 2nd hidden to next hidden to last hidden
                self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=not last_bn))  # from last hidden to out_channel
            self.mlp_bns_hidden.append(nn.BatchNorm2d(out_channel))

    def forward(self, xyz, calc_scores='softmax', bias=0):
        B, _, N, K = xyz.size()
        scores = xyz

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))
            else:
                scores = self.mlp_convs_nohidden(scores)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden) - 1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))
                    else:
                        scores = conv(scores)
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1) + bias  # B*m*N*K, where bias may bring larger gradient
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores) + bias  # B*m*N*K
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores
