import numpy as np
import open3d as o3d
import cv2


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

# 将体素网格可视化为图像
voxel_grid_image = np.asarray(voxel_grid.to_image())

# 保存体素图像
cv2.imwrite("voxel_grid_image.png", voxel_grid_image)

# # 进行体素化
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
#
# # 将体素网格可视化为点云
# # voxel_cloud = voxel_grid.to_legacy_point_cloud()
# voxel_cloud = voxel_grid.sample_points_poisson_disk(5000)
#
# # 生成一个图像
# visualizer = o3d.visualization.Visualizer()
# visualizer.create_window()
# visualizer.add_geometry(voxel_cloud)
# visualizer.get_render_option().point_size = 1
# visualizer.get_render_option().background_color = np.asarray([1, 1, 1])
# visualizer.capture_screen_image("voxel_grid_image.png")
# visualizer.destroy_window()
