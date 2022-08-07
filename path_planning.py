import os
import argparse
import math
import copy

# Basics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Point cloud and OctoMap
import laspy
import open3d as o3d
import open3d.visualization.rendering as rendering
import octomap
from utils.octomap_utils import update_freespace_by_subtraction, occupied_to_obstacles
from registration import register_points, draw_registration_result

from utils.io import import_laz_to_o3d_filter

from thirdparty.rrt_algorithms.src.rrt.rrt_star import RRTStar
from thirdparty.rrt_algorithms.src.search_space.search_space import SearchSpace
from thirdparty.rrt_algorithms.src.utilities.plotting import Plot


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('filename_tls', type=str, help='Name of LAS/LAZ file')
    parser.add_argument('filename_als', type=str, help='Name of LAS/LAZ file')
    parser.add_argument(
        '--root', type=str, default='/mnt/c/Users/aisl/Documents/dataset/', help='Name of LAS/LAZ file')
    parser.add_argument(
        '--radius-thresh', type=float, default=0.5, help='Threshold of the radius size of a tree')
    parser.add_argument(
        '--breast-height', type=float, default=1.3, help='Breast height')
    parser.add_argument(
        '--octomap-resolution', type=float, default=0.5, help='Breast height')

    return parser.parse_args()


def main():
    """ Main function
    """
    args = get_arguments()

    o3d_points_tls, mean_tls = import_laz_to_o3d_filter(
        os.path.join(args.root, args.filename_tls),
        voxel_size=0.1,
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )
    o3d_points_als, _ = import_laz_to_o3d_filter(
        os.path.join(args.root, args.filename_als),
        # offset=mean_tls,
        voxel_size=0.1,
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )

    voxel_size = 0.1
    voxel_size_down = 0.6
    reg_result = register_points(
        o3d_points_als, o3d_points_tls,
        voxel_size=voxel_size, voxel_size_down=voxel_size_down)

    o3d_points_als.transform(reg_result.transformation)

    octree_tls = octomap.OcTree(args.octomap_resolution)
    print("Insert TLS points to octree")
    octree_tls.insertPointCloud(
        pointcloud=np.asarray(o3d_points_tls.points),
        origin=np.array([0, 0, 20], dtype=float),
        # origin=mean_tls,
        maxrange=-1,
        lazy_eval=True,
    )
    print(np.asarray(o3d_points_tls.points).shape)

    octree_als = octomap.OcTree(args.octomap_resolution)
    print("Insert ALS points to octree")
    octree_als.insertPointCloud(
        pointcloud=np.asarray(o3d_points_als.points),
        origin=np.array([0, 0, 20], dtype=float),
        maxrange=-1,
        lazy_eval=True,
    )

    aabb_tls_min, aabb_tls_max = octree_tls.getMetricMin(), octree_tls.getMetricMax()
    aabb_als_min, aabb_als_max = octree_als.getMetricMin(), octree_als.getMetricMax()
    # aabb_max =
    print(aabb_tls_min, aabb_tls_max)
    print(aabb_als_min, aabb_als_max)

    update_freespace_by_subtraction(
        octree=octree_tls, aabb_max=aabb_tls_max, aabb_min=aabb_tls_min,
        resolution=args.octomap_resolution)
    occupied_tls, empty_tls = octree_tls.extractPointCloud()

    update_freespace_by_subtraction(
        octree=octree_als, aabb_max=aabb_als_max, aabb_min=aabb_als_min,
        resolution=args.octomap_resolution)
    occupied_als, empty_als = octree_als.extractPointCloud()

    # Path planning by RRT
    ## Define the search space
    X_dimensions = np.array(
        [(aabb_tls_min[0], aabb_tls_max[0]),
        (aabb_tls_min[1], aabb_tls_max[1]),
        (aabb_tls_min[1], aabb_tls_max[1])])  # dimensions of Search Space

    ## Add occupied voxels as obstacles
    ## in the format "np.array([(x_min, y_min, z_min, x_max, y_max, z_max), ...])""
    # obstacles = np.array(
    #     [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
    #     (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)]) // 10
    obstacles = occupied_to_obstacles(occupied_tls, args.octomap_resolution)
    x_init = (21, 20, 4)
    x_goal = (19, -7, 23)

    Q = np.array([(0.5, 10), (1, 20), (3, 30), (7, 30)])  # length of tree edges
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 65536  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    X = SearchSpace(X_dimensions, obstacles)
    ## Initialize the object 
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)

    ## Solve
    path = rrt.rrt_star()
    print(path)

    points, indices = [], []
    for i, p in enumerate(path):
        points.append(p)

        if i != 0:
            indices.append([i-1, i])

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(indices)

    # plot
    # plot = Plot("rrt_star_tls")
    # plot.plot_tree(X, rrt.trees)
    # if path is not None:
    #     plot.plot_path(X, path)
    # # plot.plot_obstacles(X, obstacles)
    # plot.plot_start(X, x_init)
    # plot.plot_goal(X, x_goal)
    # plot.draw(auto_open=True)

    # Visualize
    # o3d_points_tls.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=np.asarray(o3d_points_tls.points).shape))
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_points_tls,
    #                                                             voxel_size=args.octomap_resolution)
    o3d.visualization.draw_geometries([o3d_points_als, lineset])


if __name__ == '__main__':
    main()
