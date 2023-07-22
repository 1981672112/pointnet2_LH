""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True


def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True


cv2.namedWindow('show3d')
cv2.moveWindow('show3d', 0, 0)
cv2.setMouseCallback('show3d', onmouse)

dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'), '.')


def showpoints(xyz, c_gt=None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(0, 0, 0), normalizecolor=True, ballradius=10):
    global showsz, mousex, mousey, zoom, changed
    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz
    if c_gt is None:
        c0 = np.zeros((len(xyz),), dtype='float32') + 255
        c1 = np.zeros((len(xyz),), dtype='float32') + 255
        c2 = np.zeros((len(xyz),), dtype='float32') + 255
    else:
        c0 = c_gt[:, 0]  # 取一列
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]

    if normalizecolor:  # T
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    show = np.zeros((showsz, showsz, 3), dtype='uint8')  # 800 800 3   0

    def render():
        rotmat = np.eye(3)
        if not freezerot:  # T
            xangle = (mousey - 0.5) * np.pi * 1.2  # 0
        else:
            xangle = 0

        rotmat = rotmat.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2  # 0
        else:
            yangle = 0

        rotmat = rotmat.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        rotmat *= zoom  # 缩放
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]  # 2500 3 3 3

        ixyz = nxyz.astype('int32')
        show[:] = background  # 800 800 3 每个都是255 255 255
        dll.render_ball(
            ct.c_int(show.shape[0]),  # 窗口的高度。
            ct.c_int(show.shape[1]),  # 窗口的宽度。
            show.ctypes.data_as(ct.c_void_p),  # 是一个指针，指向显示窗口数据的起始位置，即像素矩阵。
            ct.c_int(ixyz.shape[0]),  # 示点云中点的数量。
            ixyz.ctypes.data_as(ct.c_void_p),  # )是一个指针，指向点云数据的起始位置。
            c0.ctypes.data_as(ct.c_void_p),  # 颜色数据的指针，分别表示R、G、B三个通道的颜色
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)  # 是球的半径。
        )

        if magnifyBlue > 0:  # F
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))
        if showrot:  # F
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)), (30, showsz - 30), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)), (30, showsz - 50), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))

    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)

        if waittime == 0:
            cmd = cv2.waitKey(10) % 256  # 无键输入返回-1 -1%256=255# % 256是对这个ASCII码取模，限制结果在0-255之间。
        else:
            cmd = cv2.waitKey(waittime) % 256

        if cmd == ord('q'):  # ord() 函数返回字符的 ASCII 码值。
            break
        elif cmd == ord('Q'):
            sys.exit(0)

        if cmd == ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_gt[:, 0]
                    c1 = c_gt[:, 1]
                    c2 = c_gt[:, 2]
            else:
                if c_pred is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_pred[:, 0]
                    c1 = c_pred[:, 1]
                    c2 = c_pred[:, 2]
            if normalizecolor:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0
            c0 = np.require(c0, 'float32', 'C')
            c1 = np.require(c1, 'float32', 'C')
            c2 = np.require(c2, 'float32', 'C')
            changed = True

        if cmd == ord('n'):  # 放大
            zoom *= 1.1
            changed = True
        elif cmd == ord('m'):  # 缩小
            zoom /= 1.1
            changed = True
        elif cmd == ord('r'):  # 还原初始
            zoom = 1.0
            changed = True
        elif cmd == ord('s'):
            cv2.imwrite('show3d.png', show)
        if waittime != 0:
            break
    return cmd


if __name__ == '__main__':
    import os
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/shapenet', help='dataset path')
    parser.add_argument('--category', type=str, default='Airplane', help='select category')
    parser.add_argument('--npoints', type=int, default=2500, help='resample points number')
    parser.add_argument('--ballradius', type=int, default=10, help='ballradius')
    opt = parser.parse_args()
    '''
    Airplane	02691156
    Bag	        02773838
    Cap	        02954340
    Car	        02958343
    Chair	    03001627
    Earphone	03261776
    Guitar	    03467517
    Knife	    03624134
    Lamp	    03636649
    Laptop	    03642806
    Motorbike   03790512
    Mug	        03797390
    Pistol	    03948459
    Rocket	    04099429
    Skateboard  04225987
    Table	    04379243'''

    cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],  # 红
                     [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],  # 绿
                     [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],  # 蓝
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # 粉
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # '/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/visualizer'
    ROOT_DIR = os.path.dirname(BASE_DIR)  # '/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master'
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

    from data_utils.ShapeNetDataLoader import PartNormalDataset

    root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    dataset = PartNormalDataset(root=root, npoints=2048, split='test', normal_channel=False)
    idx = np.random.randint(0, len(dataset))  # 991 <max=2874
    data = dataset[idx]
    point_set, _, seg = data  # data 是一个元组（point_set,cls,seg）这三个元素都是ndarray类型 [2048 3][15][47 48]

    choice = np.random.choice(point_set.shape[0], opt.npoints, replace=True)  # 是2048个点选2500点 可以重复
    point_set, seg = point_set[choice, :], seg[choice]

    seg = seg - seg.min()  # 0 1 2 3
    # print(seg,len(seg)) # [1 3 1 ... 0 1 1] 2500
    gt = cmap[seg, :]  # 只选前面的几行
    # print(gt,len(gt)) # 验证
    pred = cmap[seg, :]
    showpoints(point_set, gt, c_pred=pred, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(255, 255, 255), normalizecolor=True, ballradius=opt.ballradius)
