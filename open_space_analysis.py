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
from utils.octomap_utils import update_freespace_by_subtraction, visualize

from utils.io import import_laz_to_o3d_filter

# TODO: Remove the process requiring `imgviz`
import imgviz

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

    data = imgviz.data.arc2017()
    camera_info = data["camera_info"]
    K = np.array(camera_info["K"]).reshape(3, 3)

    o3d_points_tls, mean_tls = import_laz_to_o3d_filter(
        os.path.join(args.root, args.filename_tls),
        voxel_size=-0.1, 
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )
    o3d_points_als, _ = import_laz_to_o3d_filter(
        os.path.join(args.root, args.filename_als),
        offset=mean_tls,
        voxel_size=-0.1, 
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )

    # octree_tls = octomap.OcTree(args.octomap_resolution)
    # print("Insert TLS points to octree")
    # octree_tls.insertPointCloud(
    #     pointcloud=np.asarray(o3d_points_tls.points),
    #     origin=np.array([0, 0, 20], dtype=float),
    #     # origin=mean_tls,
    #     maxrange=-1,
    # )

    # print(np.asarray(o3d_points_tls.points).shape)
    # octree_als = octomap.OcTree(args.octomap_resolution)
    # print("Insert ALS points to octree")
    # octree_als.insertPointCloud(
    #     pointcloud=np.asarray(o3d_points_als.points),
    #     origin=np.array([0, 0, 20], dtype=float),
    #     maxrange=-1,
    # )

    # aabb_tls_min, aabb_tls_max = octree_tls.getMetricMin(), octree_tls.getMetricMax()
    # aabb_als_min, aabb_als_max = octree_als.getMetricMin(), octree_als.getMetricMax()
    # # aabb_max = 
    # print(aabb_tls_min, aabb_tls_max)
    # # print(aabb_als_min, aabb_als_max)

    # update_freespace_by_subtraction(
    #     octree=octree_tls, aabb_max=aabb_tls_max, aabb_min=aabb_tls_min,
    #     resolution=args.octomap_resolution)
    # occupied_tls, empty_tls = octree_tls.extractPointCloud()

    # update_freespace_by_subtraction(
    #     octree=octree_als, aabb_max=aabb_als_max, aabb_min=aabb_als_min,
    #     resolution=args.octomap_resolution)
    # occupied_als, empty_als = octree_als.extractPointCloud()

    # print(occupied_tls.shape, empty_tls.shape)
    # print(occupied_als.shape, empty_als.shape)

    # visualize(
    #     occupied=occupied,
    #     empty=empty,
    #     K=K,
    #     width=camera_info["width"],
    #     height=camera_info["height"],
    #     resolution=args.octomap_resolution,
    #     aabb=(aabb_min, aabb_max),
    # )

if __name__ == '__main__':
    main()
