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

    # data = imgviz.data.arc2017()
    # camera_info = data["camera_info"]
    # K = np.array(camera_info["K"]).reshape(3, 3)
    K  = np.array([[517.61981689,   0.,         317.96018872],
                  [  0.,         517.77970108, 242.81595627],
                  [  0.,           0.,           1.        ]])
    height = 480
    width = 640

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

    # octree_tls = octomap.OcTree(args.octomap_resolution)
    octree_tls = octomap.SemanticOcTree(0.5)
    print("Insert TLS points to octree")
    octree_tls.insertPointCloudAndSemantics(
        pointcloud=np.asarray(o3d_points_tls.points),
        origin=np.array([0, 0, 20], dtype=float),
        id=0,
        category=0,
        confidence=0.7,
        # origin=mean_tls,
        maxrange=-1,
        lazy_eval=True,
    )
    print(np.asarray(o3d_points_tls.points).shape)

    octree_als = octomap.SemanticOcTree(args.octomap_resolution)
    print("Insert ALS points to octree")
    octree_als.insertPointCloudAndSemantics(
        pointcloud=np.asarray(o3d_points_als.points),
        origin=np.array([0, 0, 20], dtype=float),
        id=0,
        category=0,
        confidence=0.7,
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

    print(occupied_tls.shape, empty_tls.shape)
    print(occupied_als.shape, empty_als.shape)

    TP, TN, FP, FN = calculate_metrics(
        octree_tls, octree_als, aabb_tls_min, aabb_tls_max, args.octomap_resolution)

    print(TP, TN, FP, FN)
    print("accuracy: {}".format((TP + TN) / (TP + TN + FP + FN)))
    print("precision: {}".format((TP) / (TP + FP)))
    print("recall: {}".format((TP) / (TP + FN)))
    # draw_registration_result(
    #     o3d_points_als, o3d_points_tls, reg_result.transformation)
    visualize(
        occupied=occupied_tls,
        empty=empty_tls,
        K=K,
        width=width,
        height=height,
        resolution=args.octomap_resolution,
        aabb=(aabb_tls_min, aabb_tls_max),
    )


if __name__ == '__main__':
    main()
