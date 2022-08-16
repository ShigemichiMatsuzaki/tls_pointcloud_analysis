import os
import argparse
import time

# Basics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Point cloud and OctoMap
import open3d as o3d
import cv2

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

    # o3d_points_tls, mean_tls = import_laz_to_o3d_filter(
    #     os.path.join(args.root, args.filename_tls),
    #     voxel_size=0.1,
    #     chunked_read=True,
    #     use_statistical_filter=True,
    #     nb_neighbors=10,
    #     std_ratio=15.0
    # )
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

    cv2.imwrite("watershed_seg.png", tps.chm_segmenter.vis_img)


    # for t in tps.trees: 
    #     t.classify_points(visualize=True)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    opt.point_show_normal = True

    t_accum = o3d.geometry.PointCloud()
    for t in tps.trees: 
        t_accum += t.get_points()

    # viewer.add_geometry(tps[10].stem_points)
    viewer.add_geometry(t_accum)
    viewer.run()

    viewer.destroy_window()

    # o3d.visualization.draw_geometries([o3d_points_als])

    # for t in tps.trees:
    #     print(t.points)


if __name__ == '__main__':
    main()
