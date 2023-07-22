"""
nn.init.kaiming_normal_(torch.empty(self.m2, i2, o2), nonlinearity='relu')

    这段代码使用 PyTorch 中的 nn.init.kaiming_normal_() 方法对一个形状为 (self.m2, i2, o2) 的张量进行 Kaiming He 初始化。

    具体来说，Kaiming He 初始化是一种针对深度神经网络的 初始化方法，旨在解决神经网络层数增加时出现的梯度消失和梯度爆炸问题。
    该方法根据网络层数和非线性函数的特点，动态地调整 权重的标准差，使得每个神经元的 输出的方差 保持不变。

    在 PyTorch 中，nn.init.kaiming_normal_() 方法接受两个参数：
        tensor：要初始化的张量，可以是任何形状的张量。
        nonlinearity：非线性激活函数的类型。这个参数是可选的，默认值是 'leaky_relu'。

        如果使用了 'relu' 非线性函数，则应该将 nonlinearity 参数设为 'relu'，以便在初始化时使用正确的标准差调整系数。
        在这段代码中，我们可以看到一个形状为 (self.m2, i2, o2) 的张量被初始化为一个随机的正态分布，
        其中 self.m2 是一个整数，表示张量的数量；
        i2 和 o2 分别是输入和输出的维度。
        这个初始化过程会在原地修改输入张量，所以无需返回任何值。
"""

"""
Embed PAConv into PointNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.PAConv_util import get_scorenet_input, knn, feat_trans_pointnet, ScoreNet
from cuda_lib.functional import assign_score_withk_halfkernel as assemble_pointnet


class PAConv(nn.Module):
    def __init__(self, args):
        super(PAConv, self).__init__()
        self.args = args
        self.k = args.get('k_neighbors', 20)
        self.calc_scores = args.get('calc_scores', 'softmax')

        self.m2, self.m3, self.m4 = args.get('num_matrices', [8, 8, 8])
        self.scorenet2 = ScoreNet(6, self.m2, hidden_unit=[16])  # (self, in_channel, out_channel, hidden_unit=[16], last_bn=False)
        self.scorenet3 = ScoreNet(6, self.m3, hidden_unit=[16])
        self.scorenet4 = ScoreNet(6, self.m4, hidden_unit=[16])

        i2 = 64  # channel dim of output_1st and input_2nd    前一个输入输出是后一个输入
        o2 = i3 = 64  # channel dim of output_2st and input_3rd
        o3 = i4 = 64  # channel dim of output_3rd and input_4th
        o4 = 128  # channel dim of output_4th and input_5th

        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2, self.m2 * o2)
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, i3, o3), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i3, self.m3 * o3)
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, i4, o4), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4, self.m4 * o4)

        # convolutional weight matrices in Weight Bank:
        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)  # tensor2 传入Tensor类型参数，requires_grad默认值为True，表示可训练，
        self.matrice3 = nn.Parameter(tensor3, requires_grad=True)
        self.matrice4 = nn.Parameter(tensor4, requires_grad=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 40)

    def forward(self, x, label=None, criterion=None):
        batch_size = x.size(0)  # 2
        idx, _ = knn(x, k=self.k)  # get the idx of knn in 3D space : b,n,k 2 1024 30
        # 得到knn邻居点的索引 _是两个点之间的距离
        xyz = get_scorenet_input(x, k=self.k, idx=idx)  # ScoreNet input: 3D coord difference : b,6,n,k

        x = self.conv1(x)  # 2 3>64 1024
        x = F.relu(self.bn1(x))
        ##################
        # replace the intermediate 3 MLP layers with PAConv:
        """CUDA implementation of PAConv: (presented in the supplementary material of the paper)"""
        """feature transformation:"""
        x = feat_trans_pointnet(point_input=x, kernel=self.matrice2, m=self.m2)  # b,n,m1,o1
        score2 = self.scorenet2(xyz, calc_scores=self.calc_scores, bias=0)
        """assemble with scores:"""
        x = assemble_pointnet(score=score2, point_input=x, knn_idx=idx, aggregate='sum')  # b,o1,n
        x = F.relu(self.bn2(x))

        x = feat_trans_pointnet(point_input=x, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_pointnet(score=score3, point_input=x, knn_idx=idx, aggregate='sum')
        x = F.relu(self.bn3(x))

        x = feat_trans_pointnet(point_input=x, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_pointnet(score=score4, point_input=x, knn_idx=idx, aggregate='sum')
        x = F.relu(self.bn4(x))
        ##################
        x = self.conv5(x)
        x = F.relu(self.bn5(x))

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        if criterion is not None:
            return x, criterion(x, label)
        else:
            return x


if __name__ == '__main__':
    import argparse
    from util.util import cal_loss, IOStream, load_cfg_from_cfg_file, merge_cfg_from_list


    def get_parser():
        parser = argparse.ArgumentParser(description='3D Object Classification')
        path = '/home/lh/point_cloud/test5_PAconv/PAConv-main/obj_cls/config/pointnet_paconv_train.yaml'
        parser.add_argument('--config', type=str, default=path, help='config file')
        parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None, nargs=argparse.REMAINDER)
        args = parser.parse_args()
        assert args.config is not None
        cfg = load_cfg_from_cfg_file(args.config)

        if args.opts is not None:
            cfg = merge_cfg_from_list(cfg, args.opts)

        cfg['manual_seed'] = cfg.get('manual_seed', 0)
        cfg['workers'] = cfg.get('workers', 6)
        return cfg


    args = get_parser()  # args是个config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # g = g.to(device)
    # model = model.to(device)

    input = torch.randn(2, 3, 1024).to(device)
    model = PAConv(args).to(device)
    out = model(input)
    print(out.shape, type(out))  # torch.Size([2, 40]) <class 'torch.Tensor'>

    # print(out.shape())  # TypeError: 'torch.Size' object is not callable
    # print(out.size)  # <built-in method size of Tensor object at 0x7f5b85c350a0>
    # print(out.size())  # torch.Size([2, 40])
