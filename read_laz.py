import os
import laspy as pylas
import open3d as o3d
import numpy as np
import argparse
import math

from utils.filters import pass_through_filter
# from utils.io import import_laz_to_o3d
from utils.io import import_laz_to_o3d_filter

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('filename', type=str, help='Name of LAS/LAZ file')

    return parser.parse_args()


def main():
    """ Main function
    """
    args = get_arguments()
    file_name = args.filename
    print(file_name)

    o3d_points, _ = import_laz_to_o3d_filter(
        args.filename,
        voxel_size=-0.1, 
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )

    o3d.visualization.draw_geometries([o3d_points])


if __name__ == '__main__':
    main()
