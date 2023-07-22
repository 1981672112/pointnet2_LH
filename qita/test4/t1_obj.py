# 您可以使用以下函数输出 .obj 文件，并使用 MeshLab 将其打开。
import torch


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def get_indice_pairs(p2v_map, counts, new_p2v_map, new_counts, downsample_idx, batch, xyz, window_size, i):
    # p2v_map: [n, k]
    # counts: [n, ]

    n, k = p2v_map.shape
    mask = torch.arange(k).unsqueeze(0) < counts.unsqueeze(-1)  # [n, k]
    mask_mat = (mask.unsqueeze(-1) & mask.unsqueeze(-2))  # [n, k, k]
    index_0 = p2v_map.unsqueeze(-1).expand(-1, -1, k)[mask_mat]  # [M, ]
    index_1 = p2v_map.unsqueeze(1).expand(-1, k, -1)[mask_mat]  # [M, ]

    downsample_mask = torch.zeros_like(batch).bool()  # [N, ]
    downsample_mask[downsample_idx.long()] = True

    downsample_mask = downsample_mask[new_p2v_map]  # [n, k]
    n, k = new_p2v_map.shape
    mask = torch.arange(k).unsqueeze(0).cuda() < new_counts.unsqueeze(-1)  # [n, k]
    downsample_mask = downsample_mask & mask
    mask_mat = (mask.unsqueeze(-1) & downsample_mask.unsqueeze(-2))  # [n, k, k]
    xyz_min = xyz.min(0)[0]
    if i % 2 == 0:
        window_coord = (xyz[new_p2v_map] - xyz_min) // window_size  # [n, k, 3]
    else:
        window_coord = (xyz[new_p2v_map] + 1 / 2 * window_size - xyz_min) // window_size  # [n, k, 3]

    mask_mat_prev = (window_coord.unsqueeze(2) != window_coord.unsqueeze(1)).any(-1)  # [n, k, k]
    mask_mat = mask_mat & mask_mat_prev  # [n, k, k]

    new_index_0 = new_p2v_map.unsqueeze(-1).expand(-1, -1, k)[mask_mat]  # [M, ]
    new_index_1 = new_p2v_map.unsqueeze(1).expand(-1, k, -1)[mask_mat]  # [M, ]

    index_0 = torch.cat([index_0, new_index_0], 0)
    index_1 = torch.cat([index_1, new_index_1], 0)
    return index_0, index_1


"""
p2v_map=t 7055 53
counts=t 7055
new_p2v_map=1693 255
new_counts=1693
downsample_idx 16295
batch, 130347
xyz, 130347
window_size,3
i=int 0
"""
if __name__ == '__main__':
    p2v_map = torch.randn(7055, 53)
    counts = torch.randn(7055)
    new_p2v_map = torch.randn(1693, 255)
    new_counts = torch.randn(1693)
    downsample_idx = torch.randn(16395)
    batch = torch.randn(130347)
    xyz = torch.randn(130347)
    window_size = torch.randn(3)
    i = int(0)

    a1, a2, = get_indice_pairs(p2v_map, counts, new_p2v_map, new_counts, downsample_idx, batch, xyz, window_size, i)

    #