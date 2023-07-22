"""
1. 重新排列Tensor的各个维度
2. 组合并缩减Tensor的某些维度 这里的'mean'指定池化方式。
3. 复制并扩展Tensor
4. parse_args()
"""

import argparse

from einops import rearrange, reduce, repeat
import torch

# 1. 重新排列Tensor的各个维度
# input_tensor = torch.ones(2, 2, 3)
# print(input_tensor)
# output_tensor = rearrange(input_tensor, 't b c -> b c t')
# print(output_tensor)
# # 2. 组合并缩减Tensor的某些维度
# output_tensor = reduce(input_tensor, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=2, w2=2)
# # 3. 复制并扩展Tensor
# output_tensor = repeat(input_tensor, 'h w -> h w c', c=3)


'''
这就是高级用法了，把中间维度看作r×p，然后给出p的数值，这样系统会自动把中间那个维度拆解成3×3。
这样就完成了[3, 9, 9] -> [3, 3, 3, 9]的维度转换。
这个功能就不是pytorch的内置功能可比的。
'''


# a = torch.randn(2, 6, 6)  # [3, 9, 9]
# output = rearrange(a, 'c (r p) w -> c r p w', p=3)
# print(output.shape)  # [3, 3, 3, 9]
# print(output)  # [3, 3, 3, 9]

# 1. 重新排列Tensor的各个维度
# a1 = torch.arange(1, 25).view(2, 4, 3)
# output = rearrange(a1, 'c (r p) w -> c r p w', p=2)
# print(a1)
# print(output)
# print(output.shape)

# 2. 组合并缩减Tensor的某些维度 这里的'mean'指定池化方式。
a2 = torch.arange(0, 1 * 1 * 6 * 6).view(1, 1, 6, 6).float()
output_tensor = reduce(a2, 'b c (h h2) (w w2) -> b h w c', 'max', h2=3, w2=2)  # 'max' 'meam'
print(a2)
print(output_tensor)

# a21 = torch.arange(0, 12).view(2, 2, 3).float()  # 2 2 3
# output_tensor21 = reduce(a21, 'c (h h2) (w w2) -> c h w ', 'max', h2=2, w2=1)  # 'max' 'meam'
# print(a21)
# print(output_tensor21)
#
# a22 = torch.arange(0, 12).view(2, 2, 3).float()  # 2 2 3
# output_tensor22 = reduce(a22, 'c (h h2) (w w2) -> c h w', 'max', h2=2, w2=3)  # 'max' 'meam'
# print(a22)
# print(output_tensor22)

# a22 = torch.arange(0, 12).view(2, 2, 3).float()  # 2 2 3
# output_tensor22 = reduce(a22, 'c (h h2) (w w2) -> c h w ', 'max', h2=1, w2=3)  # 'max' 'meam' 每一批 的每一行的行取最大
# print(a22)
# print(output_tensor22)
#
# a23 = torch.arange(0, 12).view(2, 2, 3).float()  # 2 2 3
# output_tensor23 = reduce(a23, 'c (h h2) (w w2) -> c h w ', 'max', h2=2, w2=1)  # 'max' 'meam' 每一批 的每一列的列取最大
# print(a23)
# print(output_tensor23)

# a23 = torch.arange(0, 72).view(2, 6, 6).float()  # 2 2 3 块池化   2行3列选一个最大的
# output_tensor23 = reduce(a23, 'c (h h2) (w w2) -> c h w', 'max', h2=2, w2=3)  # 'max' 'meam'
# print(a23)
# print(output_tensor23)

# 3. 复制并扩展Tensor
# a3 = torch.randn(2, 3)  # [9, 9]
# output_tensor = repeat(a3, 'h w -> c h w', c=3)  # 323
# output_tensor1 = repeat(a3, 'h w -> h w c', c=3)  # 233
# print(a3)
# print(output_tensor)
# print(output_tensor1)

# 4. parse_args()
# def parse_args():
#     parser = argparse.ArgumentParser('Model')
#     parser.add_argument('--model', type=str, default='pointnet2_part_seg_ssg_xiugai', help='model name')
#     parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
#     parser.add_argument('--epoch', default=150, type=int, help='epoch to run')
#     parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
#     parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
#     parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
#     parser.add_argument('--log_dir', type=str, default=None, help='log path')
#     parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
#     parser.add_argument('--npoint', type=int, default=2048, help='point Number')
#     parser.add_argument('--normal', action='store_true', default=True, help='use normals')
#     parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
#     parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
#
#     return parser.parse_args()


# def main(args):
#     print(args.epoch)
#     print(args.model)


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
