from dataloader.paths import PathsDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class CoreSetSelector:

    def __init__(self, dataset, lmdb_handle, base_size, batch_size):
        self.lmdb_handle = lmdb_handle
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset
    
    def _updated_distances(self, cluster_centers, features, min_distances):
        x = features[cluster_centers, :]
        dist = pairwise_distances(features, x, metric='euclidean')
        if min_distances is None:
            return np.min(dist, axis=1).reshape(-1, 1)
        else:
            return np.minimum(min_distances, dist)

    def _select_batch(self, features, selected_indices, N):
        new_batch = []
        min_distances = self._updated_distances(selected_indices, features, None)
        for _ in range(N):
            ind = np.argmax(min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in selected_indices
            min_distances = self._updated_distances([ind], features, min_distances)
            new_batch.append(ind)

        print('Maximum distance from cluster centers is %0.5f' % max(min_distances))
        return new_batch
    
    def select_next_batch(self, model, training_set, selection_count):
        combined_paths = training_set.image_path_subset + training_set.remaining_image_paths
        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, combined_paths), batch_size=self.batch_size, shuffle=False, num_workers=0)
        FEATURE_DIM = 2432
        average_pool_kernel_size = (32, 32)
        
        features = np.zeros((len(combined_paths), FEATURE_DIM))
        model.eval()
        model.set_return_features(True)

        average_pool_stride = average_pool_kernel_size[0] // 2
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(loader)):
                _, features_batch = model(sample['image'].cuda())
                features_batch = F.avg_pool2d(features_batch, average_pool_kernel_size, average_pool_stride)
                for feature_idx in range(features_batch.shape[0]):
                    features[batch_idx * self.batch_size + feature_idx, :] = features_batch[feature_idx, :, :, :].cpu().numpy().flatten()
                

        model.set_return_features(False)
        selected_indices = self._select_batch(features, list(range(len(training_set.image_path_subset))), selection_count)
        training_set.expand_training_set([combined_paths[i] for i in selected_indices])
