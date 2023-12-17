import numpy as np
import open3d as o3d
import cv2
# import open3d_tutorial as o3dtut
import t as o3dtut

pcd = o3d.io.read_point_cloud("/home/lh/兔子/bunny.txt", format='xyz')
print(pcd)
print(pcd.points)

print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    N = 3000
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(N)
    # Fit to unit cube.
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
              center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1,
                                                              size=(N, 3)))
    print('Displaying input point cloud ...')
    o3d.visualization.draw_geometries([pcd])

    print('Displaying voxel grid ...')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])
