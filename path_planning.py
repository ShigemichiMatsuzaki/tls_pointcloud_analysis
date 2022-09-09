import os
import argparse
import math
import copy

# Basics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union

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

def generate_candidate_points_from_free_voxels(
    free_voxels: np.ndarray, octree: Union[octomap.OcTree, octomap.SemanticOcTree],
    x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None,
)->np.ndarray:
    """Get candidate points for start/goal of path planning

    Parameters
    ----------
    free_voxels: `numpy.ndarray`
        Numpy array storing voxel locations n x 3
    octree: Union[octomap.OcTree, octomap.SemanticOcTree]
        OcTree
    x_min: `float`
        Minimum value of x
    x_max: `float`
        Maximum value of x
    y_min: `float`
        Minimum value of y
    y_max: `float`
        Maximum value of y
    z_min: `float`
        Minimum value of z
    z_max: `float`
        Maximum value of z

    Returns
    -------
    candidate_points: `numpy.ndarray`
        Numpy array storing candidates
   
    """
    candidate_points = []

    xyz_min, xyz_max = octree.getMetricMin(), octree.getMetricMax()
    if x_max is None:
        x_max = xyz_max[0]
    if y_max is None:
        y_max = xyz_max[1]
    if z_max is None:
        z_max = xyz_max[2]
    if x_min is None:
        x_min = xyz_min[0]
    if y_min is None:
        y_min = xyz_min[1]
    if z_min is None:
        z_min = xyz_min[2]

    print(xyz_min, xyz_max)

    for p in free_voxels:
        node = octree.search(p, depth=0)
        if (p[0] > x_max) or (p[0] < x_min) or (p[1] > y_max) or (p[1] < y_min) or (p[2] > z_max) or (p[2] < z_min):
            continue

        candidate_points.append(p)

        try:
            if node.getOccupancy() > 0.5:
                continue

            candidate_points.append(p)
        except:
            continue

    print("candidate point num: {}".format(len(candidate_points)))
    return np.array(candidate_points)


def is_path_feasible(
    octree: Union[octomap.OcTree, octomap.SemanticOcTree],
    path: list
) -> bool:
    """Check the feasibility of the path in the given voxel map

    Parameters
    ----------
    octree: `Union[octomap.OcTree, octomap.SemanticOcTree]`
        Voxel map
    path: `list`
        List of tuples of 3D point coordinate (x, y, z)

    Returns
    -------
    is_feasible: `bool`
        True if the path is feasible, i.e., it does not collide with any occupied voxels
    
    """
    for i in range(len(path)-1):
        if not is_line_feasible(octree, path[i], path[i+1]):
            return False

    return True


def is_line_feasible(
    octree: Union[octomap.OcTree, octomap.SemanticOcTree],
    start_point: tuple,
    end_point: tuple
) -> bool:
    """Check the feasibility of the line in the given voxel map

    Parameters
    ----------
    octree: `Union[octomap.OcTree, octomap.SemanticOcTree]`
        Voxel map
    start_point: `tuple`
        Tuple of 3D point coordinate (x, y, z) of the start point of the line
    end_point: `tuple`
        Tuple of 3D point coordinate (x, y, z) of the end point of the line

    Returns
    -------
    is_feasible: `bool`
        True if the line is feasible, i.e., it does not collide with any occupied voxels
    
    """
    np_start_point = np.asarray(start_point, dtype=np.float64)
    np_end_point = np.asarray(end_point, dtype=np.float64)

    # Unit vector from the start to the end
    v = np_end_point - np_start_point
    v /= np.linalg.norm(v)

    end = np.array([0,0,0], dtype=np.float64)
    octree.castRay(np_start_point, v, end)

    print(end, np_end_point)

    return True


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

    Q = np.array([(0.5, 30), (1, 40), (3, 50), (7, 50), (10, 30)])  # length of tree edges
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
    lineset.paint_uniform_color(np.asarray([[0], [1], [1]]))

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
    # o3d.visualization.draw_geometries([o3d_points_als, lineset])

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()


    # for t in tps.trees: 
    # viewer.add_geometry(tps[10].stem_points)
    viewer.add_geometry(o3d_points_als)
    viewer.add_geometry(lineset)

    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_show_normal = False
    opt.line_width = 50.0


    viewer.run()
    viewer.destroy_window()



if __name__ == '__main__':
    main()
