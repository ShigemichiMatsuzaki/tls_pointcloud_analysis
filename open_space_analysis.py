import os
import argparse

# Basics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# Point cloud and OctoMap
import laspy
import open3d as o3d
import open3d.visualization.rendering as rendering
import octomap
from utils.octomap_utils import update_freespace_by_subtraction, visualize, calculate_metrics, occupied_to_obstacles, filter_free_voxels
# from utils.octomap_utils import update_freespace_by_subtraction, visualize, calculate_metrics
from registration import register_points, draw_registration_result

from utils.io import import_laz_to_o3d_filter

# Path planning
from thirdparty.rrt_algorithms.src.rrt.rrt_star import RRTStar
from thirdparty.rrt_algorithms.src.search_space.search_space import SearchSpace
from path_planning import is_line_feasible, is_path_feasible, generate_candidate_points_from_free_voxels

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

    valid_ids = ["1002", "1005", "1007", "1012", "1014", "1052", "1054"]
    tls_files = []
    als_files = []
    for i in range(len(tls_files_tmp)):
        id = tls_files_tmp[i].rsplit('/', 1)[1].split('_', 1)[0]
        if id in valid_ids:
            tls_files.append(tls_files_tmp[i])
            als_files.append(als_files_tmp[i])

    print(tls_files)
    print(als_files)


    with open('open_space_analysis_result.txt', 'w+') as f:
        for i in range(len(tls_files)):

            o3d_points_tls, mean_tls = import_laz_to_o3d_filter(
                #os.path.join(args.root, args.filename_tls),
                tls_files[i],
                voxel_size=0.1,
                chunked_read=True,
                use_statistical_filter=True,
                nb_neighbors=30,
                std_ratio=10.0,
            )
            o3d_points_als, _ = import_laz_to_o3d_filter(
                # os.path.join(args.root, args.filename_als),
                # offset=mean_tls,
                als_files[i],
                voxel_size=0.1,
                chunked_read=True,
                use_statistical_filter=True,
                nb_neighbors=30,
                std_ratio=10.0,
            )

            voxel_size = 0.1
            voxel_size_down = 0.6
            reg_result = register_points(
                o3d_points_als, o3d_points_tls,
                voxel_size=voxel_size, voxel_size_down=voxel_size_down,
                robust_kernel=args.registration_robust_kernel,
                robust_thresh=args.registration_robust_threshold)

            if args.visualize:
                draw_registration_result(o3d_points_als, o3d_points_tls, reg_result.transformation)

            o3d_points_als.transform(reg_result.transformation)

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
            # aabb_max =
            # print(aabb_tls_min, aabb_tls_max)
            # print(aabb_als_min, aabb_als_max)

            update_freespace_by_subtraction(
                octree=octree_tls, aabb_max=aabb_tls_max, aabb_min=aabb_tls_min,
                resolution=args.octomap_resolution)
            occupied_tls, empty_tls = octree_tls.extractPointCloud()

            update_freespace_by_subtraction(
                octree=octree_als, aabb_max=aabb_als_max, aabb_min=aabb_als_min,
                resolution=args.octomap_resolution)
            occupied_als, empty_als = octree_als.extractPointCloud()

            # print("Resolution: {}".format(octree_als.getResolution()))

            # print(occupied_tls.shape, empty_tls.shape)
            # print(occupied_als.shape, empty_als.shape)

            TP, TN, FP, FN = calculate_metrics(
                octree_tls, octree_als, aabb_tls_min, aabb_tls_max, args.octomap_resolution)

            acc = (TP + TN) / (TP + TN + FP + FN)
            pre = (TP) / (TP + FP)
            rec = (TP) / (TP + FN)
            f.write("{} {} {} {} {} {} {}\n".format(TP, TN, FP, FN, acc, pre, rec))
            print("{} {} {} {} {} {} {}".format(TP, TN, FP, FN, acc, pre, rec))

            if not args.path_planning:
                continue

            print("Let's plan the path")
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
            while len(path_list) < 1:
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

            
            # Feasibility check on TLS
            count = 0
            for p in path_list:
                if is_path_feasible(octree_tls, p,):
                    count += 1

            visualize(
                occupied=occupied_als, #occupied_als,
                empty=occupied_tls,
                K=K,
                width=width,
                height=height,
                resolution=args.octomap_resolution,
                aabb=(aabb_als_min, aabb_als_max),
                colors=None,#occupied_als_color,
                path=path_list[0],
                # candidate_points=candidate_points,
            )


if __name__ == '__main__':
    main()
