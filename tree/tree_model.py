import numpy as np
import open3d as o3d
import math
import cv2

from utils.chm_tree_segmentation import CHMSegmenter
from utils.filters import pass_through_filter
from utils.ransac import ransac_cylinder, find_model_inliers

class TreeModel(object):
    """Class representing a tree"""

    def __init__(
        self, 
        o3d_points: o3d.geometry.PointCloud,
        classify_on_init: bool=False, 
        breast_height: float=1.5
    ):
        self.o3d_points = o3d_points
        self.is_initialized = False
        self.radius = 0.10
        self.normal_angle_tresh = 5 # [rad]
        self.breast_height = breast_height

        if classify_on_init:
            self.initialize()

        # Classified points
        self.stem_points = None
        self.non_stem_points = None
        self.bottom_point = np.array([0., 0.])

        # Metrics

    def initialize(self):
        """Initialize a tree model (calculate normals, classes, measurements etc.?)

        Parameters
        ----------
        points : numpy.ndarray
            Points of a cluster representing a tree

        """
        self.classify_points()
        self.is_initialized = True

#    def classify_points(self, visualize=False):
#        """
#        
#        """
#        print("Classify points")
#        if self.o3d_points is None:
#            raise ValueError
#
#        # Calculate normal for each point
#        self.o3d_points.estimate_normals(
#            o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius * 2, max_nn=20))
#        self.o3d_points.normalize_normals()
#
#        # Classify the points by the angle of the normal
#        vertical_axis = np.array([0.0, 0.0, 1.0])
#        dot_scores = np.dot(np.asarray(self.o3d_points.normals), vertical_axis)
#
#        self.stem_points = self.o3d_points.select_by_index(np.where(np.abs(dot_scores) < 0.10)[0], invert=False)
#        self.non_stem_points = self.o3d_points.select_by_index(np.where(np.abs(dot_scores) < 0.10)[0], invert=True)
#
#        # RANSAC-based cylinder fitting on 'stem_points'
#        ## Extract points within a certain height range
#        z_min = np.asarray(self.stem_points.points).min(axis=0)[2]
#        breast_height_point = pass_through_filter(
#            {"x": [-math.inf, math.inf],
#             "y": [-math.inf, math.inf],
#             "z": [z_min + self.breast_height, z_min + self.breast_height + 10.2]},
#             self.stem_points)
#
#        if np.asarray(breast_height_point.points).size < 10:
#            return
#
#        ## DBCAN clustering
#        cluster_labels = np.array(
#            breast_height_point.cluster_dbscan(eps=0.2, min_points=10))
#
#        ## Cylinder fitting on each cluster
#        for l in range(cluster_labels.max()+1):
#        # for cluster in clusters:
#            params, _ = ransac_cylinder(
#                np.asarray(breast_height_point.points)[cluster_labels == l],
#                num_iter=100
#            )
#
#            results = find_model_inliers(
#                np.asarray(self.o3d_points.points),
#                model='cylinder',
#                params=params
#            )
#
#            o3d_inlier_points = o3d.geometry.PointCloud()
#            o3d_inlier_points.points = o3d.utility.Vector3dVector(
#                np.asarray(self.o3d_points.points)[results["inliers_indices"]])
#
#            # o3d.visualization.draw_geometries([breast_height_point])
#            if visualize:
#                o3d.visualization.draw_geometries([o3d_inlier_points])
#        
#        # Paint stem and non-stem parts in different colors


    def get_metrics(self):
        """Get tree metrics"""

        pass

    def get_points(self):
        return self.o3d_points


class TreePointSegmener(object):
    """Class to segment map points into individual trees using CHM"""

    def __init__(
        self,
        o3d_points: o3d.geometry.PointCloud,
        chm_segmenter: CHMSegmenter,
        breast_height: float=2.5
    ):
        
        """Initialize the class with o3d points and CHMSegmenter
        
        """
        self.o3d_points = o3d_points
        self.chm_segmenter = chm_segmenter
        self.trees = []

        # Parameters
        self.radius = 0.10
        self.normal_angle_tresh = 5 # [rad]
        self.breast_height = breast_height

    def segment_trees(
        self,
        o3d_points: o3d.geometry.PointCloud,
        visualize=False):
        """Classify given points into individual trees

        Parameters
        ----------
        o3d_ponits: `open3d.geometry.PointCloud`
            Point cloud to segment
        visualize: `bool`
            `True` to visualize the segmented points during the segmentation process

        Returns
        -------
        trees: `list`
            List of `TreeModel` objects
        """
        print("segment_trees")
        trees = []

        if o3d_points is None:
            raise ValueError

        # Calculate normal for each point
        o3d_points.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius * 2, max_nn=20))
        o3d_points.normalize_normals()

        # Classify the points by the angle of the normal
        vertical_axis = np.array([0.0, 0.0, 1.0])
        dot_scores = np.dot(np.asarray(o3d_points.normals), vertical_axis)

        # Filter out ground regions (nearly vertical normals)
        # o3d_points = o3d_points.select_by_index(np.where(np.abs(dot_scores) > 0.80)[0], invert=True)
        # o3d.visualization.draw_geometries([o3d_points])
        # Filter out points with non-horizontal normals
        stem_candidate_points = o3d_points.select_by_index(np.where(np.abs(dot_scores) < 0.30)[0], invert=False)
        # non_stem_points = self.o3d_points.select_by_index(np.where(np.abs(dot_scores) < 0.10)[0], invert=True)

        #
        # RANSAC-based cylinder fitting on 'stem_points'
        #

        ## Extract points within a certain height range
        # z_min = np.asarray(stem_candidate_points.points).min(axis=0)[2]
        z_min = np.asarray(o3d_points.points).min(axis=0)[2]
        print("z_min = {}".format(z_min))
        breast_height_point = pass_through_filter(
            {"x": [-math.inf, math.inf],
             "y": [-math.inf, math.inf],
             "z": [z_min + self.breast_height-0.3, z_min + self.breast_height + 0.3]},
             stem_candidate_points)

        breast_height_point.paint_uniform_color([0, 0, 1])
        stem_candidate_points.paint_uniform_color([0, 1, 0])

        # If there are not enough points, quit
        if np.asarray(breast_height_point.points).size < 10:
            print("Don't have enough points")
            self.trees = []
            return self.trees


        ## DBCAN clustering to cluster the points that potentially include
        #  multiple tree trunks into individual trees
        cluster_labels = np.array(
            breast_height_point.cluster_dbscan(eps=0.2, min_points=5))
        print(cluster_labels)

        o3d_points.paint_uniform_color(np.array([72,134,74]) / 255)
        ## Cylinder fitting on each cluster
        for l in range(cluster_labels.max()+1):
            print("Cluster {}".format(l))
        # for cluster in clusters:
            params, _ = ransac_cylinder(
                np.asarray(breast_height_point.points)[cluster_labels == l],
                num_iter=100
            )

            results = find_model_inliers(
                np.asarray(o3d_points.points),
                model='cylinder',
                params=params,
                thresh=0.1
            )

            o3d_inlier_points = o3d.geometry.PointCloud()
            o3d_inlier_points.points = o3d.utility.Vector3dVector(
                np.asarray(o3d_points.points)[results["inlier_indices"]])
            colors = np.asarray(o3d_points.colors)
            colors[results["inlier_indices"]] = np.array([255,255,0]) / 255

            o3d_points.colors = o3d.utility.Vector3dVector(colors)

        # stem color: 239,210,30
        # leaf color: 72,134,74
        if visualize:
            o3d.visualization.draw_geometries([o3d_points])


    def do_segmentation(self):
        """Segmentation of trees
        
        """
        # Get CHM segmentation
        #  "markers": Markers yielded by watershed segmentation
        chm_seg = self.chm_segmenter.do_segmentation(window_size=5)["markers"]

        # For each segment, extract points and segment 
        for n in np.unique(chm_seg):
            ## Passthrough filtering by the bounding box of the segment
            if n == -1:
                continue 

            # Roughly crop the point using a bounding box
            seg_region_indices = np.asarray(chm_seg==n).nonzero()
            x_min = seg_region_indices[0].min()
            x_max = seg_region_indices[0].max()
            y_min = seg_region_indices[1].min()
            y_max = seg_region_indices[1].max()

            x_min, y_min = self.chm_segmenter.pixel_to_map((x_min, y_min))
            x_max, y_max = self.chm_segmenter.pixel_to_map((x_max, y_max))

            x_diff = x_max - x_min
            y_diff = y_max - y_min
            if x_diff > 15 or x_diff < 1 or y_diff > 15 or y_diff < 1:
                continue

            print(n, x_min, x_max, y_min, y_max)
            margin = 0.2
            bbx_point = pass_through_filter(
                {"x": [x_min-margin, x_max+margin],
                 "y": [y_min-margin, y_max+margin],
                 "z": [-math.inf, math.inf]}, self.o3d_points)

            # Extract points within the segment
            bbx_point = self.chm_segmenter.crop_points_by_contour(n, bbx_point)

            if bbx_point.is_empty():
                continue

            self.segment_trees(bbx_point, visualize=True)

            # tree = TreeModel(bbx_point, classify_on_init=False)
            # self.trees.append(tree)

            ## Check each point whether it's within the segment
        
    def __getitem__(self, index):
        """
        
        """
        if index > len(self.trees):
            raise IndexError

        return self.trees[index]
