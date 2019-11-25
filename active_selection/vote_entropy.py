from dataloader.paths import PathsDataset
from utils.misc import turn_on_dropout, visualize_entropy
import constants
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, defaultdict


class VoteEntropySelector:

    def __init__(self, dataset, lmdb_handle, base_size, batch_size, num_classes, soft_mode=False):
        self.lmdb_handle = lmdb_handle
        self.num_classes = num_classes
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset
        if soft_mode:
            self.batch_entropy_func = self._get_soft_vote_entropy_for_batch
        else:
            self.batch_entropy_func = self._get_vote_entropy_for_batch

    def _get_vote_entropy_for_batch(self, model, image_batch, label_batch, superpixels=None):

        outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
        
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                outputs[:, step, :, :] = torch.argmax(model(image_batch), dim=1)

        scores = []
        for i in range(image_batch.shape[0]):
            entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)
            for c in range(self.num_classes):
                p = torch.sum(outputs[i, :, :, :] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS
                entropy_map = entropy_map - (p * torch.log2(p + 1e-12))
            entropy_map[label_batch[i, :, :] == 255] = 0
        
            if not superpixels is None:
                score = entropy_map.cpu().numpy()
                unique_superpixels_as_list = np.unique(superpixels[i, :, :]).tolist()
                score_per_superpixel = defaultdict(int)
                for spx_id in unique_superpixels_as_list:
                    spx_mean = score[superpixels[i, :, :] == spx_id].mean()

                    score_per_superpixel[spx_id] = spx_mean
                    score[superpixels[i, :, :] == spx_id] = spx_mean
                scores.append(score_per_superpixel)
            else:
                scores.append(entropy_map.cpu().numpy())

        del outputs
        torch.cuda.empty_cache()

        return scores

    def _get_soft_vote_entropy_for_batch(self, model, image_batch, label_batch, superpixels=None):

        outputs = torch.cuda.FloatTensor(image_batch.shape[0], self.num_classes, image_batch.shape[2], image_batch.shape[3]).fill_(0)
        softmax = torch.nn.Softmax2d()
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                outputs[:, :, :, :] += softmax(model(image_batch))

        outputs /= constants.MC_STEPS
        scores = []
        for i in range(image_batch.shape[0]):
            entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)
            for c in range(self.num_classes):
                entropy_map = entropy_map - (outputs[i, c, :, :] * torch.log2(outputs[i, c, :, :] + 1e-12))
            entropy_map[label_batch[i, :, :] == 255] = 0

            if not superpixels is None:
                score = entropy_map.cpu().numpy()
                unique_superpixels_as_list = np.unique(superpixels[i, :, :]).tolist()
                score_per_superpixel = defaultdict(int)
                for spx_id in unique_superpixels_as_list:
                    spx_mean = score[superpixels[i, :, :] == spx_id].mean()
                    score_per_superpixel[spx_id] = spx_mean
                    score[superpixels[i, :, :] == spx_id] = spx_mean
                scores.append(score_per_superpixel)
            else:
                scores.append(entropy_map.cpu().numpy())

        del outputs
        torch.cuda.empty_cache()

        return scores

    def calculate_scores(self, model, paths, return_score_maps=False):
        model.eval()
        model.apply(turn_on_dropout)
        scores = []

        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, paths), batch_size=self.batch_size, shuffle=False, num_workers=0)

        for sample in tqdm(loader, desc='Entropy'):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            if return_score_maps:
                scores.extend(self.batch_entropy_func(model, image_batch, label_batch))
            else:
                scores.extend([x.sum() for x in self.batch_entropy_func(model, image_batch, label_batch)])

        model.eval()
        return scores


    def select_next_batch_paths(self, model, training_set, selection_count):
        
        scores = self.calculate_scores(model, training_set.remaining_image_paths)
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        model.eval()
        return selected_samples
        
    def select_next_batch(self, model, training_set, selection_count):
        training_set.expand_training_set(self.select_next_batch_paths(model, training_set, selection_count))
