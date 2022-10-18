import numpy as np
import networkx as nx
import cv2
import os
import glob
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d

from utils.chm_tree_segmentation import CHMSegmenter
from utils.io import import_laz_to_o3d_filter
from utils.visualization import visualize
from utils.filters import pass_through_filter

import teaserpp_python

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('--plot-id', type=str, help='Name of LAS/LAZ file')
    parser.add_argument(
        '--root', type=str, default='/mnt/c/Users/aisl/Documents/dataset/', help='Name of LAS/LAZ file')
    parser.add_argument(
        '--radius-thresh', type=float, default=0.5, help='Threshold of the radius size of a tree')
    parser.add_argument(
        '--breast-height', type=float, default=1.3, help='Breast height')
    parser.add_argument(
        '--octomap-resolution', type=float, default=0.5, help='Breast height')
    parser.add_argument(
        '--registration-robust-kernel', type=str, choices=['tukey', 'huber', 'none'],
        default="tukey", help='Robust kernel to be used in GICP')
    parser.add_argument(
        '--registration-robust-threshold', type=float, default=0.3, help='Robust kernel threshold to be used in GICP')
    parser.add_argument(
        '--path-planning', action='store_true',
        default=False, help='Robust kernel threshold to be used in GICP')
    parser.add_argument(
        '--visualize', action='store_true',
        default=False, help='Robust kernel threshold to be used in GICP')

    return parser.parse_args()

class CHMCorrespondenceMatcher(object):
    def __init__(
        self, 
        offset_src: np.ndarray, 
        offset_tgt: np.ndarray,
        length_per_pixel: float = 0.2,
        consistency_threshold: float = 5, 
    ):
        """Initialize CHMCorrespondenceMatcher with a consistency criterion

        Parameters
        ----------
        offset_src: `numpy.ndarray`
        offset_tgt: `numpy.ndarray`
        consistency_threshold: `float`
            Threshold of the difference of point distances to be considered consistent 

        """
        self.candidate_list = []
        self.point_set_src = None
        self.point_set_tgt = None

        self.consistency_threshold = consistency_threshold
        self.offset_src = offset_src
        self.offset_tgt = offset_tgt
        self.length_per_pixel = length_per_pixel

    def correspondence_matching_main(
        self, 
        markers_src: np.ndarray, 
        chm_src: np.ndarray, 
        markers_tgt: np.ndarray, 
        chm_tgt: np.ndarray, 
    ) -> None:
        """
        
        """
        self.point_set_src = self.chm_markers_to_points(
            markers_src, chm_src, self.offset_src)
        self.point_set_tgt = self.chm_markers_to_points(
            markers_tgt, chm_tgt, self.offset_tgt)

        o3d_src = o3d.geometry.PointCloud()
        o3d_src.points = o3d.utility.Vector3dVector(self.point_set_src)
        o3d_src.paint_uniform_color([1, 0.706, 0])
        o3d_tgt = o3d.geometry.PointCloud()
        o3d_tgt.points = o3d.utility.Vector3dVector(self.point_set_tgt)
        o3d_tgt.paint_uniform_color([0, 0.651, 0.929])

        points_min = np.asarray(o3d_src.points).min(axis=0)
        points_max = np.asarray(o3d_src.points).max(axis=0)

        dic = {"x": [-np.inf, np.inf],
               "y": [-np.inf, np.inf],
               "z": [(points_max[2] + points_min[2])/2, points_max[2]]}
        o3d_src_reg = pass_through_filter(dic, o3d_src)
        o3d_tgt_reg = pass_through_filter(dic, o3d_tgt)

        # np.asarray(o3d_src_reg.points)[:,2] = 0
        # np.asarray(o3d_tgt_reg.points)[:,2] = 0

        from registration import local_registration
        result = local_registration(
            o3d_src_reg, 
            o3d_tgt_reg, 
            np.identity(4, dtype=np.float64), 
            icp_type='point2point', 
            distance_threshold=1.0,
            voxel_size=0.1, 
            robust_kernel='none', 
            robust_thresh=1.0)

#        solution = self.solve_max_clique(self.point_set_src.T, self.point_set_tgt.T)
#
#        transform = np.identity(4, dtype=np.float64)
#        transform[:3, :3] = solution.rotation
#        transform[:3, 3] = solution.translation
#
        print(result.transformation)
        o3d_src_reg.transform(result.transformation)
        # o3d.visualization.draw_geometries([o3d_src_reg, o3d_tgt_reg])
        visualize([o3d_src_reg, o3d_tgt_reg])

        # self.candidate_list = self.generate_correspondence_candidates()

        # graph = self.generate_consistency_graph()

        # print(graph.edges())

        # self.visualize_graph(graph=graph)
        
    def chm_markers_to_points(
        self, markers: np.ndarray, chm: np.ndarray, offset: np.ndarray
    ) -> np.ndarray:
        """

        """
        (total_labels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(
            markers, 4, cv2.CV_32S)

        point_list = []
        for i in range(1, total_labels):
            u = int(centroid[i][0])
            v = int(centroid[i][1])

            z = chm[v][u] / 255 * 50

            x = centroid[i][0] * self.length_per_pixel + offset[0]
            y = -centroid[i][1] * self.length_per_pixel + offset[1]
            point_list.append(np.array([x, y, z]))

        return np.asarray(point_list)

    def generate_correspondence_candidates(self) -> list:
        """

        """
        if (self.point_set_src is None) or (self.point_set_tgt is None):
            print("point_sets are not initialized")
            raise ValueError

        candidate_list = []

        for i in range(self.point_set_src.shape[0]):
            for j in range(self.point_set_tgt.shape[0]):
                candidate_list.append((i, j))

        return candidate_list

    def generate_consistency_graph(self) -> nx.Graph:
        """

        """
        print("[generate_consistency_graph] Start")
        if not self.candidate_list:
            print("'self.candidate_list' is not initialized")
            raise ValueError
        
        # graph = nx.Graph()
        graph = nx.empty_graph(len(self.candidate_list), None, nx.Graph)

        for i in tqdm(range(len(self.candidate_list) - 1)):
            for j in range(i + 1, len(self.candidate_list)):
                if self.is_consistent(i, j):
                    graph.add_edge(i, j)

        return graph

    def is_consistent(
        self, node_idx1: int, node_idx2: int
    ) -> bool:
        """

        """
        if not self.candidate_list:
            print("'self.candidate_list' is not initialized")
            raise ValueError

        # Candidate: a tuple of indices (idx_src, idx_tgt)
        cand1 = self.candidate_list[node_idx1]
        cand2 = self.candidate_list[node_idx2]

        point1_src = self.point_set_src[cand1[0]]
        point1_tgt = self.point_set_tgt[cand1[1]]
        point2_src = self.point_set_src[cand2[0]]
        point2_tgt = self.point_set_tgt[cand2[1]]

        dist_src = np.linalg.norm(point1_src - point2_src)
        dist_tgt = np.linalg.norm(point1_tgt - point2_tgt)

        # Evaluate the difference of the distance between points
        # Consistent pair should fulfill the criterion
        if np.abs(dist_src - dist_tgt) < self.consistency_threshold:
            return True
        else:
            return False

    def visualize_graph(self, graph: nx.Graph) -> None:
        """
        
        """

        subax1 = plt.subplot(111)
        nx.draw(graph)

    def visualize_correspondences(self) -> None:
        """
        
        """
        pass

    def solve_max_clique(self, src, dst):
        """
        
        """
        # Populating the parameters
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 0.20
        solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12
        print("Parameters are:", solver_params)

        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(src, dst)

        solution = solver.getSolution()

        # Print the solution
        print("Solution is:", solution)

        # Print the inliers
        scale_inliers = solver.getScaleInliers()
        scale_inliers_map = solver.getScaleInliersMap()
        translation_inliers = solver.getTranslationInliers()
        translation_inliers_map = solver.getTranslationInliersMap()

        print("=======================================")
        print("Scale inliers (TIM pairs) are:")
        print("Note: they should not include the outlier points.")
        for i in range(len(scale_inliers)):
            print(scale_inliers[i], end=',')
        print("\n=======================================")

        print("Translation inliers are:", translation_inliers)
        print("Translation inliers map is:", translation_inliers_map)

        return solution


def main():
    args = get_arguments()

    tls_files_tmp = sorted(glob.glob(os.path.join(args.root, 'Evo_TLS_2021_thinned', args.plot_id + '*.laz')))
    als_files_tmp = sorted(glob.glob(os.path.join(args.root, 'Evo_HeliALS-TW_2021_euroSDR', args.plot_id + '*.laz')))

    if len(tls_files_tmp) != len(als_files_tmp):
        print("The numbers of TLS and ALS files are different")

    valid_ids = ["1002", "1005", "1007", "1012", "1014", "1052", "1054"]
    tls_files = []
    als_files = []
    for i in range(len(tls_files_tmp)):
        id = tls_files_tmp[i].rsplit('/', 1)[1].split('_', 1)[0]
        if id in valid_ids:
            tls_files.append(tls_files_tmp[i])
            als_files.append(als_files_tmp[i])

    o3d_points_tls, mean_tls = import_laz_to_o3d_filter(
        #os.path.join(args.root, args.filename_tls),
        tls_files[0],
        voxel_size=0.1,
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=30,
        std_ratio=10.0,
    )
    # visualize(o3d_cloud=o3d_points_tls)

    o3d_points_als, _ = import_laz_to_o3d_filter(
        # os.path.join(args.root, args.filename_als),
        # offset=mean_tls,
        als_files[0],
        voxel_size=0.1,
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=10.0,
    )

    window_size = 8 
    plot_id = tls_files[0].rsplit('/', 1)[1].split('_', 1)[0]
    chm_name_tls = tls_files[0].rsplit('/', 1)[1].split('.', 1)[0] + '_CHM.png'
    offset_tls = np.asarray(o3d_points_tls.points).min(axis=0)
    chm_segmenter_tls = CHMSegmenter(
        os.path.join('CHM', chm_name_tls), offset_x=offset_tls[0], offset_y=offset_tls[1])
    chm_segmenter_tls.do_segmentation(window_size=window_size)
    cv2.imwrite(str(plot_id) + "_tls_seg.png", chm_segmenter_tls.vis_img)

    chm_name_als = als_files[0].rsplit('/', 1)[1].split('.', 1)[0] + '_CHM.png'
    offset_als = np.asarray(o3d_points_als.points).min(axis=0)
    chm_segmenter_als = CHMSegmenter(
        os.path.join('CHM', chm_name_als), offset_x=offset_als[0], offset_y=offset_als[1])
    chm_segmenter_als.do_segmentation(window_size=window_size)
    cv2.imwrite(str(plot_id) + "_als_seg.png", chm_segmenter_als.vis_img)

    matcher = CHMCorrespondenceMatcher(offset_src=offset_als, offset_tgt=offset_tls, )
    # matcher.chm_markers_to_nodes(chm_segmenter_als.sure_fg)
    matcher.correspondence_matching_main(
        chm_segmenter_als.sure_fg, chm_segmenter_als.img,
        chm_segmenter_tls.sure_fg, chm_segmenter_tls.img)

    # cv2.imshow("", chm_segmenter_als.sure_fg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__=='__main__':
    main()