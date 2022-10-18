import open3d as o3d
import numpy as np
from typing import Union

def visualize(
    input: Union[o3d.geometry.PointCloud, list], 
    point_size: int = 5
) -> None:
    """Visualize

    Parameters
    ----------
    o3d_cloud: `open3d.geometry.PointCloud`
        Open3D point cloud

    """

    # Colorize the pointcloud based on the CityScapes color palette
    # Point cloud visualizer
    #    o3d.visualization.draw_geometries([point_list])

    # Point cloud visualizer
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Carla Lidar", width=960, height=540, left=480, top=270
    )
    opt = vis.get_render_option()
    opt.background_color = [0.05, 0.05, 0.05]
    opt.point_size = point_size
    opt.show_coordinate_frame = True

    if isinstance(input, o3d.geometry.PointCloud):
        vis.add_geometry(input)
    elif isinstance(input, list):
        o3d_points = o3d.geometry.PointCloud()
        for l in input:
            o3d_points += l

        vis.add_geometry(o3d_points)

    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    vis.add_geometry(axis)

    vis.run()
    vis.destroy_window()
