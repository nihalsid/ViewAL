import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils.colormaps import map_segmentations_to_colors, map_segmentation_to_colors
import scipy.misc
import constants
import numpy as np


class TensorboardSummary:

    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=self.directory)
        return writer

    def visualize_state(self, writer, dataset, image, target, output, global_step):
        cat_tensors = []
        print(dataset)
        for k in range(min(5, image.shape[0])):
            tensor_image = torch.from_numpy(np.transpose((np.transpose(image[k].clone().cpu().numpy(), axes=[
                                            1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)), [2, 0, 1])).float().unsqueeze(0)
            target_image = map_segmentations_to_colors(torch.squeeze(target[k:k + 1], 1).detach().cpu().numpy(), dataset=dataset).float()
            output_image = map_segmentations_to_colors(torch.max(output[k:k + 1], 1)[1].detach().cpu().numpy(), dataset=dataset).float()
            cat_tensors.append(torch.cat((tensor_image, target_image, output_image), -1))
        grid_image = make_grid(torch.cat(cat_tensors, -2), 3, normalize=False, range=(0, 255))
        writer.add_image('segmentations', grid_image, global_step)
