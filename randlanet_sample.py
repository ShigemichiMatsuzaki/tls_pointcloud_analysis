import os
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.vis import Visualizer, LabelLUT
from open3d.ml.datasets import SemanticKITTI
import numpy as np
import matplotlib.pyplot as plt

from utils.io import import_laz_to_o3d_filter

import umap
import umap.plot


def segment_points(o3d_points: o3d.geometry.PointCloud()):
    """Segment points using RandLA-Net

    Parameters
    ----------
    o3d_points: `open3d.geometry.PointCloud`
        Point cloud to segment

    Returns
    -------
    data: `dict`
        Dictionary that contains entries as follows:
        "name": `str`
            Name of the data
        "points": `numpy.ndarray`
            Numpy array storing the points # n x 3
        "labels": `numpy.ndarray`
            Numpy array storing the ground truth labels # n
        "pred": `numpy.ndarray` # n
            Numpy array storing the predicted labels
        "feat": `numpy.ndarray` # n
            Numpy array storing the intermediate features

    """
    cfg_file = "../Open3D-ML/ml3d/configs/randlanet_semantickitti.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)

    model = ml3d.models.RandLANet(**cfg.model)
    # cfg.dataset['dataset_path'] = "/mnt/d/dataset/SemanticKITTI/"
    # dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=None, device="gpu", **cfg.pipeline)

    # download the weights.
    ckpt_folder = "./logs/"
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_path = ckpt_folder + "randlanet_semantickitti_202201071330utc.pth"
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"
    if not os.path.exists(ckpt_path):
        cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
        os.system(cmd)

    # load the parameters.
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    # train_split = dataset.get_split("training")
    # data = train_split.get_data(9)
    points = np.asarray(o3d_points.points)
    labels = np.ones((points.shape[0]))

    data = {'point': points, 'feat': None, 'label': labels}

    # run inference on a single example.
    # returns dict with 'predict_labels' and 'predict_scores'.
    result = pipeline.run_inference(data)

    kitti_labels = SemanticKITTI.get_label_to_names()
    print(kitti_labels)
    v = Visualizer()
    lut = LabelLUT()
    for val in sorted(kitti_labels.keys()):
        lut.add_label(kitti_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    # result['predict_labels'] += 1 # = (result['predict_labels'] + 1).astype(np.int32)
    # pred_label_r[0] = 0
    pred_label_r = (result['predict_labels'] + 1).astype(np.int32)
    # Fill "unlabeled" value because predictions have no 0 values.
    pred_label_r[0] = 0

    feat = result['predict_features']
    print(feat.reshape((feat.shape[0] * feat.shape[2] * feat.shape[3], feat.shape[1])).shape)

    data = {
        "name": 'Pred',
        "points": data['point'], # n x 3
        "labels": data['label'], # n
        "pred": pred_label_r, # n
        "feat": feat,
    }

    # v.visualize([data])

    return data


def visualize_features(feat: np.ndarray):
    """Visualize features using UMAP

    Parameters
    ----------
    feat: `numpy.ndarray`
        Features in a form of (B, C, H, W)
    
    """
    feat = feat.reshape((feat.shape[0] * feat.shape[2] * feat.shape[3], feat.shape[1]))
    mapper = umap.UMAP(verbose=True).fit(feat)
    ax = umap.plot.points(mapper)
    plt.show()


if __name__ == "__main__":
    o3d_points, _ = import_laz_to_o3d_filter(
        '/mnt/c/Users/aisl/Documents/dataset/Evo_HeliALS-TW_2021_euroSDR/1002.laz',
        voxel_size=0.2, 
        chunked_read=True,
        use_statistical_filter=True,
        nb_neighbors=10,
        std_ratio=15.0
    )

    data = segment_points(o3d_points)

    # 15: vegetation (ground)
    # 16: trunk (tree)
    o3d_points.points = o3d.utility.Vector3dVector(data['points'][data['pred'] == 15])

    print(data['pred'])

    o3d.visualization.draw_geometries([o3d_points])