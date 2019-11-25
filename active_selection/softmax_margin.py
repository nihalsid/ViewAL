import numpy as np
from dataloader.paths import PathsDataset
from utils.misc import turn_on_dropout, visualize_entropy
import constants
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class SoftmaxMarginSelector:

    def __init__(self, dataset, lmdb_handle, base_size, batch_size):
        self.lmdb_handle = lmdb_handle
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset

    def calculate_scores(self, model, paths):
        model.eval()
        scores = []

        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, paths), batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                softmax = torch.nn.Softmax2d()
                output = softmax(model(image_batch))
                for batch_idx in range(output.shape[0]):
                    most_confident_scores = torch.max(output[batch_idx, :, :].squeeze(), dim=0)[0].cpu().numpy()
                    output_numpy = output[batch_idx, :, :, :].cpu().numpy()
                    ndx = np.indices(output_numpy.shape)
                    second_most_confident_scores = output_numpy[output_numpy.argsort(0), ndx[1], ndx[2]][-2]
                    margin = most_confident_scores - second_most_confident_scores
                    margin[(label_batch[batch_idx, :, :] == 255).cpu().numpy()] = 1
                    scores.append(np.mean(margin))
                del output, margin
                torch.cuda.empty_cache()
        return scores


    def select_next_batch(self, model, training_set, selection_count):

        scores = self.calculate_scores(model, training_set.remaining_image_paths)
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=False)))[1][:selection_count]

        model.eval()
        training_set.expand_training_set(selected_samples)
