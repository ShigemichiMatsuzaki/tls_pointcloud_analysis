import os
import pylas
import open3d as o3d
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('filename', type=str, help='Name of LAS/LAZ file')
    parser.add_argument(
        '--root', type=str, default='/media/shigemichi/HDD/dataset/', help='Name of LAS/LAZ file')

    return parser.parse_args()


def preprocess_point_cloud(pcd_down, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def import_laz_to_o3d(filepath, voxel_size=0.1):
    print(filepath)
    las = pylas.read(filepath)

    np_points = np.array(
        [las['X'], las['Y'], las['Z']]) / 1000.0
    np_points = np_points.T

    mean = np_points.mean(axis=0)
    mean[2] = 0
    np_points -= mean

    del las

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(np_points)
    if voxel_size > 0:
        o3d_points = o3d_points.voxel_down_sample(voxel_size=voxel_size)

    return o3d_points


def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    return result


def pass_through_filter(dic, pcd):

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    points[:, 0]
    x_range = np.logical_and(
        points[:, 0] >= dic["x"][0], points[:, 0] <= dic["x"][1])
    y_range = np.logical_and(
        points[:, 1] >= dic["y"][0], points[:, 1] <= dic["y"][1])
    z_range = np.logical_and(
        points[:, 2] >= dic["z"][0], points[:, 2] <= dic["z"][1])

    pass_through_filter = np.logical_and(
        x_range, np.logical_and(y_range, z_range))

    pcd.points = o3d.utility.Vector3dVector(points[pass_through_filter])
    if colors.size != 0:
        pcd.colors = o3d.utility.Vector3dVector(colors[pass_through_filter])

    return pcd


def main():
    """ Main function
    """
    args = get_arguments()

    # o3d_points_2014 = import_laz_to_o3d(
    #     os.path.join(args.root, 'Evo_TLS_2014_laz', args.filename)
    # )
    o3d_points_2019 = import_laz_to_o3d(
        os.path.join(args.root, 'Evo_TLS_2019_laz', args.filename),
        voxel_size=0.1
    )

    breast_height = 2.0
    dic = {"x": [-math.inf, math.inf],
           "y": [-math.inf, math.inf],
           "z": [breast_height-0.2, breast_height+0.2]}

    o3d_points_2019 = pass_through_filter(dic, o3d_points_2019)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            o3d_points_2019.cluster_dbscan(eps=0.20, min_points=15, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(
        labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    o3d_points_2019.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # o3d_points_2014.paint_uniform_color([1, 0.706, 0])
    # o3d_points_2019.paint_uniform_color([0, 0.651, 0.929])

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    # viewer.add_geometry(o3d_points_2014)
    viewer.add_geometry(o3d_points_2019)

    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
    # o3d.visualization.draw_geometries([source_down, target_down])


if __name__ == '__main__':
    main()
