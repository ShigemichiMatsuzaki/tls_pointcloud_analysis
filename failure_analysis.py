import os
import argparse

# Basics
import numpy as np
import glob

# Point cloud and OctoMap
import octomap
from utils.octomap_utils import visualize
from path_planning import is_path_feasible
# from utils.octomap_utils import update_freespace_by_subtraction, visualize, calculate_metrics

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
        '--region', type=int,
        default=0, help='0: All, 1: bottom half, 2: upper half')

    return parser.parse_args()


def main():
    """ Main function
    """
    args = get_arguments()

    K  = np.array([[517.61981689,   0.,         317.96018872],
                  [  0.,         517.77970108, 242.81595627],
                  [  0.,           0.,           1.        ]])
    height = 480
    width = 640

    tls_files_tmp = sorted(glob.glob(os.path.join(args.root, 'Evo_TLS_2021_thinned', args.plot_id + '*.laz')))
    als_files_tmp = sorted(glob.glob(os.path.join(args.root, 'Evo_HeliALS-TW_2021_euroSDR', args.plot_id + '*.laz')))

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

    if args.region == 0:
        suffix = "all"
    elif args.region == 1:
        suffix = "bottom"
    elif args.region == 2:
        suffix = "top"
    else:
        raise ValueError


    for file_index in range(len(tls_files)):
        plot_id = tls_files[file_index].rsplit('/', 1)[1].split('_', 1)[0]

        # paths = np.load(os.path.join("paths", "{}.npy".format(plot_id)), allow_pickle=True)
        paths = np.load(os.path.join("paths", "{}_{}.npy".format(plot_id, suffix)), allow_pickle=True)

        o3d_points_tls, mean_tls = import_laz_to_o3d_filter(
            #os.path.join(args.root, args.filename_tls),
            tls_files[file_index],
            voxel_size=0.1,
            chunked_read=True,
            use_statistical_filter=True,
            nb_neighbors=30,
            std_ratio=10.0,
        )

        octree_tls = octomap.OcTree(args.octomap_resolution)
        # octree_tls = octomap.SemanticOcTree(0.5)
        print("Insert TLS points to octree")
        octree_tls.insertPointCloud(
            pointcloud=np.asarray(o3d_points_tls.points),
            origin=np.array([0, 0, 20], dtype=float),
            maxrange=-1,
            lazy_eval=True,
        )
        occupied_tls, empty_tls = octree_tls.extractPointCloud()
        print(np.asarray(o3d_points_tls.points).shape)

        o3d_points_als, mean_als = import_laz_to_o3d_filter(
            #os.path.join(args.root, args.filename_als),
            als_files[file_index],
            voxel_size=0.1,
            chunked_read=True,
            use_statistical_filter=True,
            nb_neighbors=30,
            std_ratio=10.0,
        )
        # Read the transformation
        transformation = np.load(
            os.path.join('transforms', 'transformation_{}.npy'.format(plot_id)))

        o3d_points_als.transform(transformation)

        octree_als = octomap.OcTree(args.octomap_resolution)
        # octree_als = octomap.SemanticOcTree(0.5)
        print("Insert als points to octree")
        octree_als.insertPointCloud(
            pointcloud=np.asarray(o3d_points_als.points),
            origin=np.array([0, 0, 20], dtype=float),
            maxrange=-1,
            lazy_eval=True,
        )
        occupied_als, empty_als = octree_als.extractPointCloud()

        aabb_tls_min, aabb_tls_max = octree_tls.getMetricMin(), octree_tls.getMetricMax()

        i = 0
        for p in paths:
            is_feasible, end = is_path_feasible(octree_tls, p)
            is_feasible_als, _ = is_path_feasible(octree_als, p)
            print("{}: tls: {}, als: {}, hit_point: {}".format(
                i, is_feasible, is_feasible_als, end))
            i += 1

            if not is_feasible:
                continue

            visualize(
                occupied=occupied_tls, #occupied_als,
                empty=occupied_als,
                K=K,
                width=width,
                height=height,
                resolution=args.octomap_resolution,
                aabb=(aabb_tls_min, aabb_tls_max),
                colors=None,#occupied_als_color,
                path=p,
                candidate_points=end,
            )

if __name__ == '__main__':
    main()
