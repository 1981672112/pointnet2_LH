import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 从txt文件中读取点云数据
def read_point_cloud_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = []
    for line in lines:
        x, y, z = map(float, line.strip().split())
        points.append([x, y, z])
    return np.array(points)

# 读取点云数据
point_cloud_data = read_point_cloud_from_txt("/home/lh/兔子/bunny.txt")

# 创建Open3D点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

# 体素化参数
voxel_size = 0.01  # 体素大小

# 进行体素化
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

# 获取体素网格的属性
voxels = np.asarray(voxel_grid.get_voxels())

# 计算体素的大小
voxel_size = np.array([voxel_size, voxel_size, voxel_size])

# 计算图像的大小，以便每个体素对应一个像素
image_size = (int(1.0 / voxel_size[0]), int(1.0 / voxel_size[1]))

# 创建一个空白的图像
image = np.zeros(image_size, dtype=np.uint8)

# 将体素网格中的存在的体素置为白色
for voxel in voxels:
    x, y, z = voxel
    image[int(y / voxel_size[1]), int(x / voxel_size[0])] = 255

# 可视化图像
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig('voxel_grid_image.png', bbox_inches='tight', pad_inches=0)
plt.show()
