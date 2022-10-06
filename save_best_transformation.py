import os
import argparse

# Basics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import copy

# Point cloud and OctoMap
import laspy
import open3d as o3d
import open3d.visualization.rendering as rendering
import octomap
from utils.octomap_utils import update_freespace_by_subtraction, visualize, calculate_metrics, occupied_to_obstacles, filter_free_voxels
# from utils.octomap_utils import update_freespace_by_subtraction, visualize, calculate_metrics
from registration import register_points, draw_registration_result

from utils.io import import_laz_to_o3d_filter


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('--plot-id', type=str, default='', help='Name of LAS/LAZ file')
    parser.add_argument(
        '--root', type=str, default='/mnt/c/Users/aisl/Documents/dataset/', help='Name of LAS/LAZ file')
    parser.add_argument(
        '--radius-thresh', type=float, default=0.5, help='Threshold of the radius size of a tree')
    parser.add_argument(
        '--breast-height', type=float, default=1.3, help='Breast height')
    parser.add_argument(
        '--octomap-resolution', type=float, default=0.5, help='Breast height')
    parser.add_argument(
        '--registration-robust-kernel', type=str, choices=['tukey', 'huber', 'none'],
        default="tukey", help='Robust kernel to be used in GICP')
    parser.add_argument(
        '--registration-robust-threshold', type=float, default=0.3, help='Robust kernel threshold to be used in GICP')
    parser.add_argument(
        '--path-planning', action='store_true',
        default=False, help='Robust kernel threshold to be used in GICP')
    parser.add_argument(
        '--visualize', action='store_true',
        default=False, help='Robust kernel threshold to be used in GICP')
    parser.add_argument(
        '--iteration', type=int,
        default=100, help='Robust kernel threshold to be used in GICP')

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

    tls_files_tmp = sorted(glob.glob(os.path.join(args.root, 'Evo_TLS_2021_thinned', args.plot_id + '*.laz')))
    als_files_tmp = sorted(glob.glob(os.path.join(args.root, 'Evo_HeliALS-TW_2021_euroSDR', args.plot_id + '*.laz')))

    if len(tls_files_tmp) != len(als_files_tmp):
        print("The numbers of TLS and ALS files are different")


    # valid_ids = ["1002", "1005", "1007", "1012", "1014", "1052", "1054"]
    valid_ids = ["1052"]
    tls_files = []
    als_files = []
    for i in range(len(tls_files_tmp)):
        id = tls_files_tmp[i].rsplit('/', 1)[1].split('_', 1)[0]
        if id in valid_ids:
            tls_files.append(tls_files_tmp[i])
            als_files.append(als_files_tmp[i])

    print(tls_files)
    print(als_files)

    for file_index in range(len(tls_files)):
        max_acc = 0
        best_transform = None
        o3d_points_tls_org, mean_tls = import_laz_to_o3d_filter(
            #os.path.join(args.root, args.filename_tls),
            tls_files[file_index],
            voxel_size=0.1,
            chunked_read=True,
            use_statistical_filter=True,
            nb_neighbors=30,
            std_ratio=10.0,
        )
        # o3d.visualization.draw_geometries([o3d_points_tls])

        o3d_points_als_org, _ = import_laz_to_o3d_filter(
            # os.path.join(args.root, args.filename_als),
            # offset=mean_tls,
            als_files[file_index],
            voxel_size=0.1,
            chunked_read=True,
            use_statistical_filter=True,
            nb_neighbors=10,
            std_ratio=10.0,
        )

        for i in range(10):

            plot_id = tls_files[file_index].rsplit('/', 1)[1].split('_', 1)[0]

            o3d_points_tls = copy.deepcopy(o3d_points_tls_org)
            o3d_points_als = copy.deepcopy(o3d_points_als_org)

            voxel_size = 0.1
            voxel_size_down = 0.6
            reg_result = register_points(
                o3d_points_als, o3d_points_tls,
                voxel_size=voxel_size, 
                voxel_size_down=voxel_size_down,
                registration_mode=0,
                robust_kernel=args.registration_robust_kernel,
                robust_thresh=args.registration_robust_threshold,
                z_min=0.0,
                z_max=4.0)

            if args.visualize:
                draw_registration_result(o3d_points_als, o3d_points_tls, reg_result.transformation)

            #reg_result = register_points(
            #    o3d_points_als, o3d_points_tls,
            #    voxel_size=voxel_size, 
            #    voxel_size_down=voxel_size_down,
            #    registration_mode=1,
            #    robust_kernel=args.registration_robust_kernel,
            #    robust_thresh=args.registration_robust_threshold * 0.7,
            #    z_min=0.0,
            #    z_max=4.0,
            #    init_transform=reg_result.transformation)

            #if args.visualize:
            #    draw_registration_result(o3d_points_als, o3d_points_tls, reg_result.transformation)

            o3d_points_als.transform(reg_result.transformation)
            print(reg_result.transformation)

            octree_tls = octomap.OcTree(args.octomap_resolution)
            # octree_tls = octomap.SemanticOcTree(0.5)
            print("Insert TLS points to octree")
            octree_tls.insertPointCloud(
                pointcloud=np.asarray(o3d_points_tls.points),
                origin=np.array([0, 0, 20], dtype=float),
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

            update_freespace_by_subtraction(
                octree=octree_tls,
                aabb_max=aabb_tls_max,
                aabb_min=aabb_tls_min,
                resolution=args.octomap_resolution)

            update_freespace_by_subtraction(
                octree=octree_als, 
                aabb_max=aabb_als_max, 
                aabb_min=aabb_als_min,
                resolution=args.octomap_resolution)

            TP, TN, FP, FN = calculate_metrics(
                octree_tls, octree_als, aabb_tls_min, aabb_tls_max, args.octomap_resolution)

            acc = (TP + TN) / (TP + TN + FP + FN)
            print("Accuracy: {}".format(acc))

            if acc > max_acc:
                max_acc = acc
                best_transform = reg_result.transformation
                print("  Best one!")

        np.save(
            os.path.join('transforms', 'transformation_{}.npy'.format(plot_id)), 
            best_transform,
        )


if __name__ == '__main__':
    main()
