import open3d as o3d
import numpy as np
import copy


def preprocess_point_cloud(pcd, voxel_size=-1):
    """Calculate FPFH features for the given point cloud

    """
    #print(":: Downsample with a voxel size %.3f." % voxel_size)

    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = copy.deepcopy(pcd)

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

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points[pass_through_filter])
    if colors.size != 0:
        new_pcd.colors = o3d.utility.Vector3dVector(
            colors[pass_through_filter])

    return new_pcd
