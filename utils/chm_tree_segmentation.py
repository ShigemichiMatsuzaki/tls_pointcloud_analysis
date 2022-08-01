import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
# How to generate CHM using LAStools
#   https://rapidlasso.com/2014/11/04/rasterizing-perfect-canopy-height-models-from-lidar/
# Watershed segmentation
#   https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

def find_local_maxima(img: np.ndarray, window_size: int=1, is_inverse: bool=False)->np.ndarray:
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

def chm_tree_segmentation(
    img: np.ndarray, window_size: int=7, height_thresh: int=10)->dict:
    """Segmentation of CHM via watershed segmentation 

    Parameters
    ----------
    img: `numpy.ndarray`
        Input CHM image as grayscale
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
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations = 1)

    # Extract local minima and use them as markers of segments
    # as sure foreground area
    sure_fg = find_local_maxima(img, window_size=window_size, is_inverse=False)

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

    return {"markers": markers, "sure_fg": sure_fg, "vis_img": img}

def main():
    img = cv.imread('CHM/1002_CHM.png', cv.IMREAD_GRAYSCALE)
    result = chm_tree_segmentation(img)

    cv.imwrite("watershed_seg.png", result['vis_img'])

if __name__=='__main__':
    main()