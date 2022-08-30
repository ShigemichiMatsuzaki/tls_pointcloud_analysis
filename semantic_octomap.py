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
from utils.octomap_utils import update_freespace_by_subtraction, visualize, calculate_metrics
from registration import register_points, draw_registration_result

from utils.io import import_laz_to_o3d_filter

from utils.chm_tree_segmentation import CHMSegmenter
from tree.tree_model import TreeModel, TreePointSegmener


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
        voxel_size=0.1,
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )

    # Get CHM segmentation result
    chm_name = args.filename_als.rsplit('/', 1)[1].split('.')[0] + '_CHM.png'
    offset = np.asarray(o3d_points_als.points).min(axis=0)
    chm_segmenter = CHMSegmenter(os.path.join('CHM', chm_name), offset_x=offset[0], offset_y=offset[1])

    tps = TreePointSegmener(o3d_points=o3d_points_als, chm_segmenter=chm_segmenter)
    tps.do_segmentation()

    octree_als = octomap.SemanticOcTree(args.octomap_resolution)
    print("Insert ALS points to octree")
    for t in tps:
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

    print(occupied_als.shape, empty_als.shape, occupied_als_color.shape)

    # draw_registration_result(
    #     o3d_points_als, o3d_points_tls, reg_result.transformation)
    visualize(
        occupied=occupied_als,
        empty=empty_als,
        K=K,
        width=width,
        height=height,
        resolution=args.octomap_resolution,
        aabb=(aabb_als_min, aabb_als_max),
        colors=occupied_als_color
    )


if __name__ == '__main__':
    main()
