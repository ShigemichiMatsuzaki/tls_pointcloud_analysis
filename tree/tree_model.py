import numpy as np
import open3d as o3d
import math
import cv2

from utils.chm_tree_segmentation import CHMSegmenter
from utils.filters import pass_through_filter


class TreeModel(object):
    """Class representing a tree"""

    def __init__(self, o3d_points: o3d.geometry.PointCloud, classify_on_init: bool=False):
        self.points = o3d_points
        self.is_initialized = False
        self.bottom_point = np.array([0., 0.])

        if classify_on_init:
            self.initialize()

    def initialize(
        self, 
    ):
        """Initialize a tree model (calculate normals, classes, measurements etc.?)

        Parameters
        ----------
        points : numpy.ndarray
            Points of a cluster representing a tree

        """
        pass
        self.is_initialized = True

    def classify_points(self):
        """
        
        """
        if self.points is None:
            raise ValueError

    def get_metrics(self):
        """Get tree metrics"""

        pass

    def get_points(self):
        return self.points


class TreePointSegmener(object):
    """Class to segment map points into individual trees using CHM"""

    def __init__(self, o3d_points: o3d.geometry.PointCloud, chm_segmenter: CHMSegmenter):
        """Initialize the class with o3d points and CHMSegmenter
        
        """
        self.points = o3d_points
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

            if x_max - x_min > 15 or y_max - y_min > 15:
                continue

            print(n, x_min, x_max, y_min, y_max)
            bbx_point = pass_through_filter(
                {"x": [x_min, x_max],
                 "y": [y_min, y_max],
                 "z": [-math.inf, math.inf]}, self.points)

            tree = TreeModel(bbx_point, classify_on_init=False)
            self.trees.append(tree)

            ## Check each point whether it's within the segment
