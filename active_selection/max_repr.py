from active_selection import vote_entropy
from dataloader.paths import PathsDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import math
from tqdm import tqdm


class MaxRepresentativeSelector:

    def __init__(self, dataset, lmdb_handle, base_size, batch_size, num_classes):
        self.lmdb_handle = lmdb_handle
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.vote_entropy_selector = vote_entropy.VoteEntropySelector(dataset, lmdb_handle, base_size, batch_size, num_classes, True)


    def _get_features_for_images(self, model, images):
        features = []
        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, images), batch_size=self.batch_size, shuffle=False, num_workers=0)
        model.eval()
        model.set_return_features(True)
        average_pool_kernel_size = (32, 32)
        average_pool_stride = average_pool_kernel_size[0] // 2
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(loader)):
                image_batch = sample['image'].cuda()
                _, features_batch = model(image_batch)
                for feature_idx in range(features_batch.shape[0]):
                    features.append(F.avg_pool2d(features_batch[feature_idx, :, :, :], average_pool_kernel_size,
                                                 average_pool_stride).squeeze().cpu().numpy().flatten())
        model.set_return_features(False)
        return features

    def _max_representative_samples(self, image_features, candidate_image_features, selection_count):
        all_distances = pairwise_distances(image_features, candidate_image_features, metric='euclidean')
        selected_sample_indices = []
        print('Finding max representative candidates..')
        minimum_distances = np.ones((len(image_features))) * float('inf')
        for _ in tqdm(range(selection_count)):
            current_best_score = float("-inf")
            current_best_idx = None
            current_minimum_distances = None
            for i in range(len(candidate_image_features)):
                if i not in selected_sample_indices:
                    selected_sample_indices.append(i)
                    tmp_distances = np.minimum(minimum_distances, all_distances[:, i])
                    tmp_score = np.sum(tmp_distances) * -1
                    if tmp_score > current_best_score:
                        current_best_score = tmp_score
                        current_minimum_distances = tmp_distances
                        current_best_idx = i
                    selected_sample_indices.pop()
            selected_sample_indices.append(current_best_idx)
            minimum_distances = current_minimum_distances
        return selected_sample_indices

    def select_next_batch(self, model, training_set, selection_count):
        candidate_images = self.vote_entropy_selector.select_next_batch_paths(model, training_set, selection_count * 2)
        all_image_features = self._get_features_for_images(model, training_set.image_path_subset + training_set.remaining_image_paths)
        candidate_features = self._get_features_for_images(model, candidate_images)
        selected_candidate_indices = self._max_representative_samples(all_image_features, candidate_features, selection_count)
        training_set.expand_training_set([candidate_images[i] for i in selected_candidate_indices])
