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


def draw_registration_result(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transformation) -> None:
    """Draw the registered point clouds

    Parameters
    ----------
    source: `open3d.geometry.PointCloud`
        Source point cloud
    target: `open3d.geometry.PointCloud`
        Target point cloud
    transformation: `numpy.ndarray`
        Resulting transformation

    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],)
    # zoom=0.4559,
    #front=[0.6452, -0.3036, -0.7011],
    #lookat=[1.9892, 2.0208, 1.8945],
    # up=[-0.2779, -0.9482, 0.1556])


def global_registration(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_fpfh,
        target_fpfh,
        voxel_size):
    """RANSAC-based global registration

    Parameters
    ----------
    source: `open3d.geometry.PointCloud`
        Source point cloud
    target: `open3d.geometry.PointCloud`
        Target point cloud
    source_fpfh:
        FPFH feature for the source point cloud
    target_fpfh:
        FPFH feature for the target point cloud
    voxel_size:
        Voxel size of the point cloud

    Results
    -------
    result:
        Result of the registration

    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))

    return result


def local_registration(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transformation,
        voxel_size):
    """ICP-based registration for refinement

    Parameters
    ----------
    source: `open3d.geometry.PointCloud`
        Source point cloud
    target: `open3d.geometry.PointCloud`
        Target point cloud
    transformation: `numpy.ndarray`
        Resulting transformation
    voxel_size: `float`
        Voxel size of the point cloud

    Results
    -------
    result:
        Result of the registration

    """

    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target, distance_threshold, transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())

    return result


def register_points(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size,
        voxel_size_down):
    """Register point clouds through RANSAC-based global registration
    and ICP-based local refinement.

    Parameters
    ----------
    source: `o3d.geometry.PointCloud`
        Source point cloud
    target: `o3d.geometry.PointCloud`
        Target point cloud
    voxel_size: `float`
        Size of the voxel size of the original point cloud
    voxel_size_down: `float`
        Size of the voxel size for global registration

    Return
    ------
    return: 
        Result of the registration

    """
    dic = {'x': [-math.inf, math.inf],
           'y': [-math.inf, math.inf],
           'z': [0, 5]}

    source_pt = pass_through_filter(dic, source)
    target_pt = pass_through_filter(dic, target)

    source_down, source_fpfh = preprocess_point_cloud(
        source_pt, voxel_size_down)
    target_down, target_fpfh = preprocess_point_cloud(
        target_pt, voxel_size_down)

    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_down*2, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_down*2, max_nn=30))
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    # Global registration
    result = global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size_down)

    # Local registration as refinement
    result = local_registration(
        source, target,
        result.transformation,
        # np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        voxel_size)

    return result


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

    result = register_points(
        source, target, voxel_size=0.1, voxel_size_down=0.6)

    draw_registration_result(source, target, result.transformation)


if __name__ == '__main__':
    main()
