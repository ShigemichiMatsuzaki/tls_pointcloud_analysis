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
from utils.octomap_utils import update_freespace_by_subtraction, visualize, calculate_metrics, occupied_to_obstacles, filter_free_voxels
from path_planning import is_line_feasible, is_path_feasible, generate_candidate_points_from_free_voxels
from registration import register_points, draw_registration_result

from utils.io import import_laz_to_o3d_filter

from utils.chm_tree_segmentation import CHMSegmenter
from tree.tree_model import TreeModel, TreePointSegmener

# Path planning
from thirdparty.rrt_algorithms.src.rrt.rrt_star import RRTStar
from thirdparty.rrt_algorithms.src.search_space.search_space import SearchSpace

# DL
from randlanet_sample import segment_points

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

    K  = np.array([[517.61981689,   0.,         317.96018872],
                  [  0.,         517.77970108, 242.81595627],
                  [  0.,           0.,           1.        ]])
    height, width = 480, 640

    o3d_points_als, _ = import_laz_to_o3d_filter(
        os.path.join(args.root, args.filename_als),
        # offset=mean_tls,
        voxel_size=0.10,
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )

    # Separate ground points and tree points using RandLA-Net
    # results = segment_points(o3d_points_als)

    # o3d_points_als_ground = o3d.geometry.PointCloud()
    # o3d_points_als_ground.points = o3d.utility.Vector3dVector(
    #     results['points'][results['pred'] == 15])

    # o3d.visualization.draw_geometries([o3d_points_als_ground])

    # 15: vegetation (ground)
    # 16: trunk (tree)
    o3d_points_als_tree = o3d_points_als
    # o3d_points_als_tree = o3d.geometry.PointCloud()
    # o3d_points_als_tree.points = o3d.utility.Vector3dVector(
    #     results['points'][results['pred'] == 16])
    # o3d_points_als.colors = o3d.utility.Vector3dVector(
    #     np.asarray(o3d_points_als.colors)[results['pred'] == 16])

    # Get CHM segmentation result
    chm_name = args.filename_als.rsplit('/', 1)[1].split('.')[0] + '_CHM.png'
    offset = np.asarray(o3d_points_als_tree.points).min(axis=0)
    chm_segmenter = CHMSegmenter(
        os.path.join('CHM', chm_name),
        offset_x=offset[0], offset_y=offset[1])

    tps = TreePointSegmener(
        o3d_points=o3d_points_als_tree, chm_segmenter=chm_segmenter)
    tps.do_segmentation()

    octree_als = octomap.SemanticOcTree(args.octomap_resolution)
    octree_als.insertPointCloudAndSemantics(
        pointcloud=np.asarray(o3d_points_als.points),
        origin=np.array([0, 0, 20], dtype=float),
        id=0,
        category=TreeModel.GROUND,
        confidence=0.7,
        maxrange=-1,
        lazy_eval=True,
    )

    print("Insert ALS points to octree")
    for t in tps:
        # t.visualize()
        print(t.get_points(TreeModel.STEM))
        octree_als.insertPointCloudAndSemantics(
            pointcloud=np.asarray(t.get_points(TreeModel.STEM).points),
            origin=np.array([0, 0, 20], dtype=float),
            id=0,
            category=TreeModel.STEM,
            confidence=0.7,
            maxrange=-1,
            lazy_eval=True,
        )

        octree_als.insertPointCloudAndSemantics(
            pointcloud=np.asarray(t.get_points(TreeModel.NON_STEM).points),
            origin=np.array([0, 0, 20], dtype=float),
            id=0,
            category=TreeModel.NON_STEM,
            confidence=0.7,
            maxrange=-1,
            lazy_eval=True,
        )

    aabb_als_min, aabb_als_max = octree_als.getMetricMin(), octree_als.getMetricMax()
    # aabb_max =
    print(aabb_als_min, aabb_als_max)

    update_freespace_by_subtraction(
        octree=octree_als, aabb_max=aabb_als_max, aabb_min=aabb_als_min,
        resolution=args.octomap_resolution)
    occupied_als, empty_als, occupied_als_color = octree_als.extractPointCloud()

    empty_als_filtered = filter_free_voxels(empty_als, octree_als)

    obstacles = occupied_to_obstacles(occupied_als, args.octomap_resolution)

    candidate_points = generate_candidate_points_from_free_voxels(
        empty_als, octree_als, z_min=5.0)

    Q = np.array([(0.5, 10), (1, 20), (3, 30), (7, 30)])  # length of tree edges
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1000  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    X_dimensions = np.array(
        [(aabb_als_min[0], aabb_als_max[0]),
        (aabb_als_min[1], aabb_als_max[1]),
        (aabb_als_min[1], aabb_als_max[1])])

    X = SearchSpace(X_dimensions, obstacles)

    path_list = []
    while len(path_list) < 0:
        path = None
        while path is None:
            init_index = np.random.randint(0, candidate_points.shape[0])
            goal_index = np.random.randint(0, candidate_points.shape[0])

            dist = np.linalg.norm(candidate_points[init_index] - candidate_points[goal_index])
            
            n = 0
            while (init_index == goal_index) or (dist > 40) or (dist < 10) or (n < 10):
                goal_index = np.random.randint(0, candidate_points.shape[0])
                dist = np.linalg.norm(candidate_points[init_index] - candidate_points[goal_index])

                n += 1
            
            if n == 10:
                continue

            x_init = tuple(candidate_points[init_index]) # (21, 20, 5)
            x_goal = tuple(candidate_points[goal_index]) # (19, -7, 23)

            # Initialize the object 
            rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)

            # ## Solve
            # path = None
            path = rrt.rrt_star()

            print()

        path_list.append(path)

    # draw_registration_result(
    #     o3d_points_als, o3d_points_tls, reg_result.transformation)
    print(path_list)

    """
    """

    # visualize(
    #     occupied=empty_als_filtered, #occupied_als,
    #     empty=empty_als,
    #     K=K,
    #     width=width,
    #     height=height,
    #     resolution=args.octomap_resolution,
    #     aabb=(aabb_als_min, aabb_als_max),
    #     colors= None,#occupied_als_color,
    #     path=None,
    #     # candidate_points=candidate_points,
    # )


if __name__ == '__main__':
    main()
