import cv2 as cv
import numpy as np
import open3d as o3d
from typing import Union

# How to generate CHM using LAStools
#   https://rapidlasso.com/2014/11/04/rasterizing-perfect-canopy-height-models-from-lidar/
# Watershed segmentation
#   https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

class CHMSegmenter(object):

    def __init__(self, img_name: str, length_per_pixel: float=0.2, offset_x: float=0.0, offset_y: float=0.0):
        """

        Parameters
        ----------
        image_name: `str`
            CHM image name
        length_per_pixel: `float`
            Resolution of CHM.
        offset_x: `float`
            Offset of the map in X axis
        offset_y: `float`
            Offset of the map in Y axis
        
        """
        self.img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        self.length_per_pixel = length_per_pixel
        self.offset_x, self.offset_y = offset_x, offset_y

        self.markers = None
        self.sure_fg = None
        self.vis_img = None
        self.contours = []

    def find_local_maxima(self, img: np.ndarray, window_size: int=1, is_inverse: bool=False)->np.ndarray:
        """Find local maxima in a grayscale image
        A local maximum is defined as a pixel that has a larger value 
        than any of its neighbors 

        Parameters
        ----------
        img: `numpy.ndarray`
            OpenCV image in grayscale
        window_size: `int`
            Window size around a pixel of interest to evaluate the maximum
        is_inverse: `bool`
            If `True`, `local_maxima` has 0 on local maxima and 1 in the rest.
            Otherwise, other way around.

        Returns
        -------
        local_maxima: `numpy.ndarray`
            A 2D array with the same shape as `img` indicating the local maxima pixels

        """
        local_maxima = np.ones_like(img) * 255 if is_inverse else np.zeros_like(img) 

        for i in range(window_size, img.shape[0]-window_size):
            for j in range(window_size, img.shape[1]-window_size):
                if img[i,j] == img[i-window_size:i+window_size, j-window_size:j+window_size].max() and img[i, j] != 0:
                    local_maxima[i, j] = 0 if is_inverse else 255

        return local_maxima

    def do_segmentation(
        self, window_size: int=7, height_thresh: int=10)->dict:
        """Segmentation of CHM via watershed segmentation 

        Parameters
        ----------
        window_size: `int`
            Window size of local minima detection
        height_thresh: `int`
            Threshold in binarization of CHM for estimating background

        Returns
        -------
        dict: `dict`
            Dictionary that stores the following values:
                "markers": Array with the same size as input storing marker labels
                "sure_fg": Mask image of sure foreground regions
                "vis_img": Visualization image
        """

        kernel = np.ones((3,3), np.uint8)
        # Denoising
        # img = cv.morphologyEx(self.img, cv.MORPH_CLOSE, kernel, iterations = 1)
        img = self.img

        # Extract local minima and use them as markers of segments
        # as sure foreground area
        sure_fg = self.find_local_maxima(img, window_size=window_size, is_inverse=False)

        # Extract regions being surely background
        ret, thresh = cv.threshold(img, height_thresh, 255, cv.THRESH_BINARY_INV)
        thresh = ~thresh
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)
        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=2)

        # Finding unknown region
        unknown = cv.subtract(sure_bg, sure_fg)

        # Separating each connected component
        ret, markers = cv.connectedComponents(sure_fg)
        markers += 1

        # In upcoming watershed segmentation,
        # regions of value 0 are treated as unknown regions
        # where the border is estimated via watershed.
        markers[unknown==255] = 0

        # 'cv.watershed()' expects input of 8UC3
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        markers = cv.watershed(img, markers)

        # Visualization
        img[markers == -1] = [0,0,255] # Draw borders
        img[sure_fg == 255] = [0,255,255] # Draw seed points

        # Store the calculated values in the class variables
        self.markers = markers
        self.sure_fg = sure_fg
        self.vis_img = img

        for i in np.unique(markers):
            if i == -1:
                continue
            marker_i = np.asarray(markers==i, dtype=np.uint8)
            contour, _ = cv.findContours(marker_i, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            self.contours.append(contour[0])

        return {"markers": markers, "sure_fg": sure_fg, "vis_img": img}

    def pixel_to_map(
        self, pixel: Union[list, tuple, np.array]
    ) -> Union[list, tuple, np.array]:
        """Convert pixel value in CHM to the map coordinate

        Parameter
        ---------
        pixel: `Union[list, tuple, np.array]`
            Pixel coordinate (u, v) to convert
        
        Returns
        -------
        map_coord: `Union[list, tuple, np.array]`
            Map coordinate (X, Y)
        
        """
        x = pixel[0] * self.length_per_pixel + self.offset_x
        y = pixel[1] * self.length_per_pixel + self.offset_y

        if isinstance(pixel, list):
            map_coord = [x, y]
        elif isinstance(pixel, tuple):
            map_coord = (x, y)
        else:
            map_coord = np.array([x, y])

        return map_coord

    def map_to_pixel(
        self, map_coord: Union[list, tuple, np.array]
    ) -> Union[list, tuple, np.array]:
        """Convert map coordinate to the pixel coordinate in the CHM image

        Parameter
        ---------
        map_coord: `Union[list, tuple, np.array]`
            Map coordinate (X, Y) to convert
        
        Returns
        -------
        pixel_coord: `Union[list, tuple, np.array]`
            Pixel coordinate (u, v)
        
        """
        u = int((map_coord[0] - self.offset_x) / self.length_per_pixel)
        v = int((map_coord[1] - self.offset_y) / self.length_per_pixel)

        if isinstance(map_coord, list):
            pixel_coord = [u, v]
        elif isinstance(map_coord, tuple):
            pixel_coord = (u, v)
        else:
            pixel_coord = np.array([u, v], dtype=np.int32)

        return pixel_coord


    def is_within_contour(self, contour, point_3d: np.ndarray):
        """Check if the given 3D point is within a region defined by image contour
        
        """
        img_point = self.map_to_pixel(point_3d)
        is_within = (cv.pointPolygonTest(contour, (img_point[1].item(), img_point[0].item()), False) > 0)

        return is_within


    def crop_points_by_contour(self, label_id: int, o3d_points: o3d.geometry.PointCloud):
        """
        
        """
        points = []
        for p in np.asarray(o3d_points.points):
            if self.is_within_contour(self.contours[label_id], p):
                points.append(p)

        o3d_inlier_points = o3d.geometry.PointCloud()
        if len(points) > 0:
            o3d_inlier_points.points = o3d.utility.Vector3dVector(np.array(points))

        return o3d_inlier_points


def main():

    chm_segmenter = CHMSegmenter('CHM/1002_CHM.png',)
    result = chm_segmenter.do_segmentation()

    cv.imwrite("watershed_seg.png", result['vis_img'])

if __name__=='__main__':
    main()