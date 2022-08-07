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

    def classify_points(self):
        """
        
        """
        print("Classify points")
        if self.o3d_points is None:
            raise ValueError

        # Calculate normal for each point
        self.o3d_points.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius * 2, max_nn=20))
        self.o3d_points.normalize_normals()

        # Classify the points by the angle of the normal
        vertical_axis = np.array([0.0, 0.0, 1.0])
        dot_scores = np.dot(np.asarray(self.o3d_points.normals), vertical_axis)

        self.stem_points = self.o3d_points.select_by_index(np.where(np.abs(dot_scores) < 0.10)[0], invert=False)
        self.non_stem_points = self.o3d_points.select_by_index(np.where(np.abs(dot_scores) < 0.10)[0], invert=True)

        # RANSAC-based cylinder fitting on 'stem_points'
        ## Extract points within a certain height range
        z_min = np.asarray(self.stem_points.points).min(axis=0)[2]
        breast_height_point = pass_through_filter(
            {"x": [-math.inf, math.inf],
             "y": [-math.inf, math.inf],
             "z": [z_min + self.breast_height, z_min + self.breast_height + 10.2]},
             self.stem_points)

        if np.asarray(breast_height_point.points).size < 10:
            return

        ## DBCAN clustering
        cluster_labels = np.array(
            breast_height_point.cluster_dbscan(eps=0.2, min_points=10))

        ## Cylinder fitting on each cluster
        for l in range(cluster_labels.max()+1):
        # for cluster in clusters:
            params, _ = ransac_cylinder(
                np.asarray(breast_height_point.points)[cluster_labels == l],
                num_iter=100
            )

            inliers_indices = find_model_inliers(
                np.asarray(self.o3d_points.points),
                model='cylinder',
                params=params
            )

        o3d.visualization.draw_geometries([breast_height_point])
        
        # Paint stem and non-stem parts in different colors


    def get_metrics(self):
        """Get tree metrics"""

        pass

    def get_points(self):
        return self.o3d_points


class TreePointSegmener(object):
    """Class to segment map points into individual trees using CHM"""

    def __init__(self, o3d_points: o3d.geometry.PointCloud, chm_segmenter: CHMSegmenter):
        """Initialize the class with o3d points and CHMSegmenter
        
        """
        self.o3d_points = o3d_points
        self.chm_segmenter = chm_segmenter
        self.trees = []


    def do_segmentation(self):
        """Segmentation of trees
        
        """
        # Get CHM segmentation
        #  "markers": Markers yielded by watershed segmentation
        chm_seg = self.chm_segmenter.do_segmentation()["markers"]

        # For each segment, extract points and segment 
        for n in np.unique(chm_seg):
            ## Passthrough filtering by the bounding box of the segment
            if n == -1:
                continue 

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
            bbx_point = pass_through_filter(
                {"x": [x_min, x_max],
                 "y": [y_min, y_max],
                 "z": [-math.inf, math.inf]}, self.o3d_points)

            tree = TreeModel(bbx_point, classify_on_init=False)
            self.trees.append(tree)

            ## Check each point whether it's within the segment
        
    def __getitem__(self, index):
        """
        
        """
        if index > len(self.trees):
            raise IndexError

        return self.trees[index]
