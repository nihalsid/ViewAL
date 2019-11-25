import numpy as np
import torch


def create_nyu40_label_colormap():
    return {
        255: [0, 0, 0],
        0: [190, 153, 112],
        1: [189, 198, 255],
        2: [213, 255, 0],
        3: [158, 0, 142],
        4: [32, 99, 155],#[152, 255, 82]#[255,192,0]
        5: [119, 77, 0],
        6: [60, 174, 163],#[122, 71, 130]#[32, 99, 155]
        7: [0, 174, 126],
        8: [0, 125, 181],
        9: [0, 143, 156],
        10: [107, 104, 130],
        11: [255,192,0],#[255, 229, 2]#[60, 174, 163]
        12: [117, 68, 177],
        13: [1, 255, 254],
        14: [0, 21, 68],
        15: [255, 166, 254],
        16: [194, 140, 159],
        17: [98, 14, 0],
        18: [0, 71, 84],
        19: [255, 219, 102],
        20: [0, 118, 255],
        21: [67, 0, 44],
        22: [1, 208, 255],
        23: [232, 94, 190],
        24: [145, 208, 203],
        25: [255, 147, 126],
        26: [95, 173, 78],
        27: [0, 100, 1],
        28: [255, 238, 232],
        29: [0, 155, 255],
        30: [255, 0, 86],
        31: [189, 211, 147],
        32: [133, 169, 0],
        33: [149, 0, 58],
        34: [255, 2, 157],
        35: [187, 136, 0],
        36: [0, 185, 23],
        37: [1, 0, 103],
        38: [0, 0, 255],
        39: [255, 0, 246]
    }


def get_colormap(dataset):

    if dataset.startswith('scannet') or dataset.startswith('suncg') or dataset.startswith('matterport')  or dataset.startswith('scenenet'):
        return create_nyu40_label_colormap()
    raise Exception('No colormap for dataset found')


def map_segmentations_to_colors(segmentations, dataset):
    rgb_masks = []
    for segmentation in segmentations:
        rgb_mask = map_segmentation_to_colors(segmentation, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def map_segmentation_to_colors(segmentation, dataset):
    colormap = get_colormap(dataset)
    colored_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
    for label in np.unique(segmentation).tolist():
        colored_segmentation[segmentation == label, :] = colormap[label]

    colored_segmentation /= 255.0
    return colored_segmentation
