import numpy as np
import ctypes as ct
import cv2
import sys
import os

"""
def showpoints(xyz, c_gt=None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(0, 0, 0), normalizecolor=True, ballradius=10):
这是一个用于在三维空间中可视化点云数据的函数。

它接受一个点云的坐标数组 xyz 和可选的真实值和预测值颜色数组 c_gt 和 c_pred，
可以调整窗口等待时间、是否显示旋转、是否放大蓝色通道、背景颜色等参数。
它使用了 OpenCV 库和一个名为 render_ball 的外部函数来绘制点云，还可以根据鼠标移动控制视角和缩放。
在窗口中按 'q' 键可以退出程序，按 't' 或 'p' 键可以更改点的颜色，按 'r' 键可以重置视角和缩放。

xyz: 三维点云坐标，形状为 (N, 3)，其中 N 是点的数量。
c_gt: 用于可视化的点云颜色，形状为 (N, 3)，其中 N 是点的数量。如果为 None，则所有点的颜色均为白色。
c_pred: 与 c_gt 含义相同，但是用于可视化预测结果。如果为 None，则不显示预测结果。
waittime: 在显示完点云后等待多长时间（以毫秒为单位）。
showrot: 是否在可视化时显示旋转控件。
magnifyBlue: 是否放大蓝色通道的值。当 magnifyBlue>0 时，所有蓝色通道值都会乘以 magnifyBlue。
freezerot: 当旋转控件显示时，是否冻结旋转。
background: 显示窗口的背景颜色，是一个元组，包含三个整数，分别表示红、绿、蓝通道的颜色值。
normalizecolor: 是否对颜色进行归一化。当为 True 时，颜色值将除以 255 进行归一化。
ballradius: 点云中球体的半径，用于可视化球体。
"""

# 这段代码的作用是将点云坐标标准化并将其映射到屏幕上。
# 首先，xyz矩阵减去其均值，使所有点都以原点为中心。
# 然后，通过计算所有点到原点的距离的最大值，得到点云的半径。通过将点云的半径缩放到屏幕大小的2.2倍来将点云标准化。
# 如果c_gt是None，则所有点都将用白色表示。否则，c_gt矩阵中的三个通道将被用作点的颜色，即c0，c1和c2分别表示点云中每个点的红、绿、蓝通道的颜色值。

# 导入动态链接库 目录地址
BASE_DIR = r'/home/lh/point_cloud/test1_pointnet2/Pointnet_Pointnet2_pytorch-master/visualizer'  # os.path.dirname(os.path.abspath(__file__))
showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True


# showsz：窗口展示的大小，默认为800像素。
# mousex、mousey：鼠标的x、y坐标，初始化为0.5，即窗口的中心。
# zoom：展示的缩放比例，默认为1.0，即原始大小。
# changed：布尔型变量，标记窗口是否需要更新。如果发生鼠标事件，该变量被设置为True，表示窗口需要重新绘制。

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
    xyz = xyz - xyz.mean(axis=0)  # 移动到以中心点为原点 即xyz平均值
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    # 首先计算了每个点的三个坐标的平方，然后按行(axis=-1)相加并开根号，得到了每个点距离原点的距离，也就是半径。
    # 最后取所有点的半径的最大值，作为整个点云数据的半径。这个半径将被用来缩放点云数据，使其适合于在屏幕上显示。

    xyz /= (radius * 2.2) / showsz
    # 将点云坐标除以一个常数，该常数的计算方式是点云半径的2.2倍除以显示大小（showsz）。
    # 这里通过将点云的坐标值缩放到 显示范围内，使得点云可以正确地显示在屏幕上。
    # 同时，这里的半径是通过对点云中每个点的坐标求平方和再开方得到的，用于计算点云的尺度大小。
    if c_gt is None:
        c0 = np.zeros((len(xyz),), dtype='float32') + 255
        c1 = np.zeros((len(xyz),), dtype='float32') + 255
        c2 = np.zeros((len(xyz),), dtype='float32') + 255
    else:
        c0 = c_gt[:, 0]  # 每个取一列
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]

    # 这段代码的作用是将点云坐标标准化并将其映射到屏幕上。
    # 首先，xyz矩阵减去其均值，使所有点都以原点为中心。
    # 然后，通过计算所有点到原点的距离的最大值，得到点云的半径。通过将点云的半径缩放到屏幕大小的2.2倍来将点云标准化。
    # 如果c_gt是None，则所有点都将用白色表示。
    # 否则，c_gt矩阵中的三个通道将被用作点的颜色，即c0，c1和c2分别表示点云中每个点的红、绿、蓝通道的颜色值。

    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    # 这段代码用于归一化颜色数组。如果normalizecolor参数为True，
    # 则会将 每个颜色数组除以该数组的最大值，然后乘以255，以确保每个颜色值在0到255之间。
    # 这里使用了1e-14来避免除以0的错误，因为最大值可能为0。

    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')
    # 这行代码使用 np.require 函数将 c0 数组转换为 C-contiguous array（即内存连续的数组），
    # 并将其类型转换为 float32。C-contiguous array 在内存中是连续存储的，因此在对其进行操作时可以获得更好的性能。
    # np.require 函数的三个参数分别为数组本身，所需类型和所需排列方式（例如 C、F 或 A）。

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render():
        rotmat = np.eye(3)  # 旋转单位矩阵3行
        if not freezerot:
            xangle = (mousey - 0.5) * np.pi * 1.2
            # 这行代码用于计算在x轴上旋转的角度，
            # 其中mousey表示鼠标在y轴上的位置，0.5表示画布中心位置，np.pi * 1.2表示最大旋转角度。
            # 如果freezerot为True，则xangle为0，不进行旋转。
        else:
            xangle = 0
        rotmat = rotmat.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
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
        # 这是一个OpenCV的函数，用于在窗口中展示图像。
        # 第一个参数是窗口的名称，第二个参数是需要展示的图像。
        # 函数内部会创建一个窗口并显示图像，如果同名窗口已经存在则会先销毁它再创建新的窗口。
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord('q'):
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

        if cmd == ord('n'):
            zoom *= 1.1
            changed = True
        elif cmd == ord('m'):
            zoom /= 1.1
            changed = True
        elif cmd == ord('r'):
            zoom = 1.0
            changed = True
        elif cmd == ord('s'):
            cv2.imwrite('show3d.png', show)
        if waittime != 0:
            break
    return cmd


if __name__ == '__main__':
    cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                     [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])
    np.random.seed(100)
    showpoints(np.random.randn(100, 3), c_gt=cmap, c_pred=None, waittime=0,
               showrot=False, magnifyBlue=0, freezerot=False, background=(255, 255, 255),
               normalizecolor=True, ballradius=10)
"""
dll.render_ball(,,,）
这段代码调用了名为render_ball的C++函数，并传入了多个参数：

ct.c_int(show.shape[0])和ct.c_int(show.shape[1])分别表示显示窗口的高度和宽度。
show.ctypes.data_as(ct.c_void_p)是一个指针，指向显示窗口数据的起始位置，即像素矩阵。
ct.c_int(ixyz.shape[0])表示点云中点的数量。
ixyz.ctypes.data_as(ct.c_void_p)是一个指针，指向点云数据的起始位置。
c0.ctypes.data_as(ct.c_void_p),c1.ctypes.data_as(ct.c_void_p)和c2.ctypes.data_as(ct.c_void_p)是颜色数据的指针，分别表示R、G、B三个通道的颜色值。
ct.c_int(ballradius)是球的半径。
该函数的作用是将点云数据渲染成球形，并填充到显示窗口中。
"""
