from re import A
import numpy as np
import open3d as o3d
import math
import cv2
from typing import Optional, Union

from utils.chm_tree_segmentation import CHMSegmenter
from utils.filters import pass_through_filter, cylinder_model_filter
from utils.ransac import ransac_cylinder, find_model_inliers


class TreeModel(object):
    """Class representing a tree"""

    STEM = 0
    NON_STEM = 1
    GROUND = 2

    # stem color: 239,210,30
    # leaf color: 72,134,74
    stem_color = np.array([239, 210, 30]) / 255
    non_stem_color = np.array([72, 134, 74]) / 255

    def __init__(
        self, 
        points: Optional[Union[o3d.geometry.PointCloud, dict]] = None,
        labels: Optional[np.ndarray] = None,
    ):
        """Initializatoin of the class by giving point cloud(s)

        Parameters
        ----------
        points: `Union[o3d.geometry.PointCloud, dict]`
            Points of an individual tree.
            If you provide it as `o3d.geometry.PointCloud`, also set `labels`.
            `dict` type argument should have keys 'stem' and 'non_stem'.
        labels: `Optional[np.ndarray]`
            Point-wise labels indicating the type {'stem', 'non_stem'} of each point.
            It is required if `o3d_points` is `o3d.geometry.PointCloud`.
        
        """
        assert (points is None or isinstance(points, dict) or labels is not None), \
            "When provide `o3d_points` as `o3d.geometry.PointCloud`, please also give me point labels."


        # Initialize points and labels
        if points is None:
            self.o3d_points = o3d.geometry.PointCloud()
            self.labels = np.empty(np.asarray(self.o3d_points.points).shape)
        elif isinstance(points, dict):
            # Classified points
            stem_labels = np.full(len(points['stem'].points), self.STEM, dtype=np.int32)
            non_stem_labels = np.full(len(points['non_stem'].points), self.NON_STEM, dtype=np.int32)

            points['stem'].paint_uniform_color(self.stem_color)
            points['non_stem'].paint_uniform_color(self.non_stem_color)

            self.o3d_points = points['stem'] + points['non_stem']
            self.labels = np.concatenate((stem_labels, non_stem_labels))

            # Initialize the coordinate point
            np_stem_points = np.asarray(points['stem'].points)
            self.bottom_point = np_stem_points.mean(axis=0)
            self.bottom_point[2] = np_stem_points.min(axis=0)[2]
        else:

            colors = np.asarray(points.colors)
            colors[labels == self.STEM] = np.asarray(self.stem_color)
            colors[labels == self.NON_STEM] = np.asarray(self.non_stem_color)
            points.colors = o3d.utility.Vector3dVector(colors)

            self.o3d_points = points
            self.labels = labels

    def initialize(self):
        """Initialize a tree model (calculate normals, classes, measurements etc.?)

        Parameters
        ----------
        points : numpy.ndarray
            Points of a cluster representing a tree

        """
        self.is_initialized = True

    def get_metrics(self):
        """Get tree metrics"""

        pass

    def get_points(self, label=None):
        if label is None:
            return self.o3d_points
        elif label in [self.STEM, self.NON_STEM]:
            ret_points = o3d.geometry.PointCloud()
            ret_points.points = o3d.utility.Vector3dVector(
                np.asarray(self.o3d_points.points)[self.labels == label])
            ret_points.colors = o3d.utility.Vector3dVector(
                np.asarray(self.o3d_points.colors)[self.labels == label])

            return ret_points
        else:
            raise ValueError

    def visualize(self):
        o3d.visualization.draw_geometries([self.o3d_points])


class TreePointSegmener(object):
    """Class to segment map points into individual trees using CHM"""

    def __init__(
        self,
        o3d_points: o3d.geometry.PointCloud,
        chm_segmenter: CHMSegmenter,
        breast_height: float=3.
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

    def __getitem__(self, index):
        if index > len(self.trees):
            raise IndexError

        return self.trees[index]

    def __iter__(self):
       for t in self.trees:
          yield t

    def segment_trees(
        self,
        o3d_points: o3d.geometry.PointCloud,
        visualize=False
    ) -> list:
        """Classify given points into individual trees.
        The input point cloud is expected to be roughly segmented into 
        an individual tree by CHM.

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

        # If there are not enough points, quit
        if np.asarray(breast_height_point.points).size < 7:
            print("Don't have enough points")
            return trees

        ## DBCAN clustering to cluster the points that potentially include
        #  multiple tree trunks into individual trees
        cluster_labels = np.array(
            breast_height_point.cluster_dbscan(eps=0.2, min_points=5))
        print(cluster_labels)

        ## Cylinder fitting on each cluster
        for l in range(cluster_labels.max()+1):
            print("Cluster {}".format(l))

        # for cluster in clusters:
            params, _ = ransac_cylinder(
                np.asarray(breast_height_point.points)[cluster_labels == l],
                num_iter=100
            )

            # Accumulate stem points
            results = cylinder_model_filter(
                np.asarray(o3d_points.points),
                params=params,
                mode='on',
                thresh=0.1
            )

            o3d_stem_points = o3d.geometry.PointCloud()
            o3d_stem_points.points = o3d.utility.Vector3dVector(
                np.asarray(o3d_points.points)[results["inlier_indices"]])
            
            # TODO: Likelihood of the stem should be tested
            #  - Connectivity
            #  - Size
            #  - Point distribution (deviation) around the detected stem?

            if len(o3d_stem_points.points) == 0:
                continue

            # Accumulate non-stem regions by growing regions from the stem points
            non_stem_candidate = np.asarray(o3d_points.points)[results["outlier_indices"]]
            results = cylinder_model_filter(
                non_stem_candidate,
                params=params,
                mode='within',
                thresh=2.5
            )
            non_stem_points = non_stem_candidate[results["inlier_indices"]]

            o3d_non_stem_points = o3d.geometry.PointCloud()
            o3d_non_stem_points.points = o3d.utility.Vector3dVector(non_stem_points)

            tree = TreeModel({"stem": o3d_stem_points, "non_stem": o3d_non_stem_points})

            # Append the tree model
            trees.append(tree)

            print(len(o3d_stem_points.points), len(o3d_non_stem_points.points))

        return trees


    def do_segmentation(self):
        """Segmentation of trees
        
        """
        # Get CHM segmentation
        #  "markers": Markers yielded by watershed segmentation
        # chm_seg = self.chm_segmenter.do_segmentation(window_size=5)["markers"]
        chm_seg = self.chm_segmenter.do_segmentation(window_size=5)["segments"]

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

            tmp = y_min
            y_min = y_max
            y_max = tmp

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

            # o3d.visualization.draw_geometries([bbx_point])

            trees = self.segment_trees(bbx_point, visualize=True)

            # tree = TreeModel(bbx_point, classify_on_init=False)
            self.trees += trees
            ## Check each point whether it's within the segment
        

