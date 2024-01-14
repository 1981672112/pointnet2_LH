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

# 获取体素网格的边界框
min_bound, max_bound = voxel_grid.get_axis_aligned_bounding_box()

# 计算图像的大小，以便每个体素对应一个像素
image_size = (int((max_bound[0] - min_bound[0]) / voxel_size),
              int((max_bound[1] - min_bound[1]) / voxel_size))

# 创建一个空白的图像
image = np.zeros(image_size, dtype=np.uint8)

# 获取体素中心点坐标
voxel_centers = np.asarray(voxel_grid.get_voxels()) + voxel_size / 2

# 将体素网格中的存在的体素置为白色
for x, y, z in voxel_centers:
    x_idx = int((x - min_bound[0]) / voxel_size)
    y_idx = int((y - min_bound[1]) / voxel_size)
    image[y_idx, x_idx] = 255

# 可视化图像
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig('voxel_grid_image.png', bbox_inches='tight', pad_inches=0)
plt.show()
