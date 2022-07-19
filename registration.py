import os
import argparse
import math
import copy

import laspy

import open3d as o3d
import open3d.visualization.rendering as rendering

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.ransac import ransac_cylinder
from utils.io import import_laz_to_o3d_filter
from utils.filters import preprocess_point_cloud, pass_through_filter


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('filename_tls', type=str, help='Name of LAS/LAZ file')
    parser.add_argument('filename_als', type=str, help='Name of LAS/LAZ file')
    parser.add_argument(
        '--root', type=str, default='/media/shigemichi/HDD/dataset/', help='Name of LAS/LAZ file')
    parser.add_argument(
        '--radius-thresh', type=float, default=0.5, help='Threshold of the radius size of a tree')
    parser.add_argument(
        '--breast-height', type=float, default=1.3, help='Breast height')

    return parser.parse_args()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def main():
    """ Main function
    """
    args = get_arguments()

    voxel_size = 0.6
    source, mean_tls = import_laz_to_o3d_filter(
        os.path.join(args.root, args.filename_tls),
        voxel_size=0.1, 
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )
    target, _ = import_laz_to_o3d_filter(
        os.path.join(args.root, args.filename_als),
        # offset=mean_tls,
        voxel_size=0.1, 
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )

    dic = {'x': [-math.inf, math.inf],
           'y': [-math.inf, math.inf],
           'z': [0, 5]}

    source_pt = pass_through_filter(dic, source)
    target_pt = pass_through_filter(dic, target)

    source_down, source_fpfh = preprocess_point_cloud(source_pt, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pt, voxel_size)

    print(len(source_down.points))
    print(len(target_down.points))

    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    # target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    # source_down, source_fpfh = preprocess_point_cloud(source, 0.1)
    # target_down, target_fpfh = preprocess_point_cloud(target, 0.1)

    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

    distance_threshold = voxel_size # * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    print(result.transformation)

    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target, distance_threshold, result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())

    print(len(source.points))
    print(len(target.points))

    print(result.transformation)
    draw_registration_result(source, target, result.transformation)

if __name__ == '__main__':
    main()
