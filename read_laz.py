import pylas
import open3d as o3d
import numpy as np
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('filename', type=str, help='Name of LAS/LAZ file')

    return parser.parse_args()


def main():
    """ Main function
    """
    args = get_arguments()
    file_name = args.filename
    las = pylas.read(file_name)
    print(las.points.size)
    print(las.point_format.dimension_names)

    np_points = np.array([las['X'], las['Y'], las['Z']]) / 1000.0

    del las

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(np_points.T)
    # pcd = las.points[:3, :]
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_points,
    #                                                            voxel_size=0.10)
    o3d_points = o3d_points.voxel_down_sample(voxel_size=0.1)
    o3d.visualization.draw_geometries([o3d_points])


if __name__ == '__main__':
    main()
