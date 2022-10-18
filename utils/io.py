import laspy
import open3d as o3d
import numpy as np
from tqdm import tqdm
import math
from utils.filters import pass_through_filter


def import_laz_to_o3d(filepath, voxel_size=0.1, chunked_read=True):
    print(filepath)

    # TODO: This part should be replaced with chunked reading
    # using `laspy.open()` and `LasReader.chunk_iterator()`
    if chunked_read:
        np_points = None
        chunk_size = 1_000_000
        with laspy.open(filepath) as f:
            print(f.header.point_format[3].name,
                  f.header.point_format[3].num_bits)
            with tqdm(total=f.header.point_count // chunk_size + 1) as pbar:
                for las in tqdm(f.chunk_iterator(chunk_size)):
                    # print(las.x, las.y, las.z)
                    # np_points_tmp = np.array(
                    #     [las['X'], las['Y'], las['Z']]) / 1000.0
                    np_points_tmp = np.array(
                        [las['x'], las['y'], las['z']])

                    if np_points is None:
                        np_points = np_points_tmp.T
                    else:
                        np_points = np.concatenate(
                            (np_points, np_points_tmp.T), axis=0)

    else:
        las = laspy.read(filepath)
        del las
        np_points = np.array(
            [las['X'], las['Y'], las['Z']]) / 1000.0
        np_points = np_points.T

    print("[import_laz_to_o3d] max: {}, min: {}".format(np_points.max(axis=0), np_points.min(axis=0)))

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(np_points)
    if voxel_size > 0:
        print("Voxelization. resolution: {}".format(voxel_size))
        o3d_points = o3d_points.voxel_down_sample(voxel_size=voxel_size)

    return o3d_points


def import_laz_to_o3d_filter(
        filepath,
        x_min=-math.inf,
        y_min=-math.inf,
        z_min=-math.inf,
        x_max=math.inf,
        y_max=math.inf,
        z_max=math.inf,
        offset=None,
        voxel_size=0.1,
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=15,
        std_ratio=2.0,
        color=None):
    """Load LAZ file, convert it to Open3D, and apply filters

    Parameters
    ----------
    filepath: `str`
        Name of the LAZ file to load
    x_min: `float`
        Minimum value of x axis for pass through filter
    y_min: `float`
        Minimum value of y axis for pass through filter
    z_min: `float`
        Minimum value of z axis for pass through filter
    x_max: `float`
        Maximum value of x axis for pass through filter
    y_max: `float`
        Maximum value of y axis for pass through filter
    z_max: `float`
        Maximum value of z axis for pass through filter
    offset: `numpy.ndarray`
        Value to subtract from the raw point values
    voxel_size: `float`
        Size of voxel for downsampling by voxelization
    chunked_read: `bool`
        If `True`, read the file by chunk, instead of full file
    use_statistical_filter: `bool`
        If `True`, apply statistical filter
    nb_neighbors: `float`
        The number of neighbors to consider in statistical filter
    std_ratio: `float`
        The threshold level based on the standard deviation of the average distances across the point cloud.    

    Returns
    -------
    o3d_points: `o3d.geometry.PointCloud`
        Open3D point cloud generated from the loaded LAZ file
    offset: `numpy.ndarray`
        Offset values applied to the points to shift them
    """
    print(filepath)

    o3d_points = import_laz_to_o3d(
        filepath,
        voxel_size=voxel_size,
        chunked_read=chunked_read,
    )

    # Pass through filter
    dic = {"x": [x_min, x_max],
           "y": [y_min, y_max],
           "z": [z_min, z_max]}
    o3d_points = pass_through_filter(dic, o3d_points)

    # Statistical filter
    if use_statistical_filter:
        print("Statistical filter")
        cl, ind = o3d_points.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        o3d_points = o3d_points.select_by_index(ind)

    np_points = np.asarray(o3d_points.points)
    if offset is None:
        offset = np_points.mean(axis=0)
        offset[2] = np_points[:, 2].min()

    # print(np_points)
    np_points -= offset
    # print(np_points)
    # o3d_points.points = o3d.utility.Vector3dVector(np_points)

    if isinstance(color, list) or isinstance(color, tuple):
        o3d_points.paint_uniform_color(color)

    return o3d_points, offset
