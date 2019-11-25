from dataloader.paths import PathsDataset
from dataloader import indoor_scenes
from active_selection.vote_entropy import VoteEntropySelector
from utils.misc import turn_on_dropout, visualize_entropy, visualize_spx_dataset
import constants
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, defaultdict

class RegionalVoteEntropySelector:

    def __init__(self, dataset, lmdb_handle, superpixel_dir, base_size, batch_size, num_classes, region_size, overlap_handler, mode):
        self.lmdb_handle = lmdb_handle
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.superpixel_dir = superpixel_dir
        self.overlap_handler = overlap_handler
        self.vote_entropy_selector = VoteEntropySelector(dataset, lmdb_handle, base_size, batch_size, num_classes)
        self.region_size = region_size
        if mode == 'window':
            self.select_next_batch = self.select_next_batch_with_windows
        elif mode == 'superpixel':
            self.select_next_batch = self.select_next_batch_with_superpixels
        else:
            raise NotImplementedError

    # superpixel based selection methods

    def select_next_batch_with_superpixels(self, model, training_set, selection_count):
        
        model.eval()
        model.apply(turn_on_dropout)

        loader = DataLoader(indoor_scenes.IndoorScenesWithAllInfo(self.dataset, self.lmdb_handle, self.superpixel_dir, self.base_size, training_set.all_train_paths), batch_size=self.batch_size, shuffle=False, num_workers=0)
        scores = []
        superpixel_masks = []
        
        #visualize_entropy.max_weight = 96*96
        for sample in tqdm(loader, desc='Entropy'):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            superpixel_batch = sample['superpixel']
            superpixel_masks.extend([superpixel_batch[i, :, :] for i in range(superpixel_batch.shape[0])])
            scores.extend(self.vote_entropy_selector.batch_entropy_func(model, image_batch, label_batch, superpixel_batch.numpy()))

        all_train_scenes = sorted(list(set([indoor_scenes.IndoorScenesWithAllInfo.get_scene_id_from_image_path(self.dataset, x) for x in training_set.all_train_paths])))
        scene_indices = [all_train_scenes.index(indoor_scenes.IndoorScenesWithAllInfo.get_scene_id_from_image_path(self.dataset, im_path)) for im_path in training_set.all_train_paths]

        superpixel_ids = []
        superpixel_scores_expanded = []
        
        for image_score_idx, superpixel_scores in enumerate(scores):
            for superpixel_idx in superpixel_scores.keys():
                superpixel_ids.append((scene_indices[image_score_idx], image_score_idx, superpixel_idx))
                superpixel_scores_expanded.append(superpixel_scores[superpixel_idx])

        _sorted_scores = np.array(list(list(zip(*sorted(zip(superpixel_ids, superpixel_scores_expanded), key=lambda x: x[1], reverse=True)))[0]))
        sorted_scores = np.zeros((_sorted_scores.shape[0], _sorted_scores.shape[1] + 1), dtype=np.int32)
        sorted_scores[:, 0:_sorted_scores.shape[1]] = _sorted_scores

        total_pixels_selected = 0
        selected_regions = OrderedDict()
        image_superpixels = defaultdict(list)
        ctr = 0

        print('Selecting superpixels...')
        pbar = tqdm(total=selection_count)

        while total_pixels_selected < selection_count * self.base_size[0] * self.base_size[1] and ctr < sorted_scores.shape[0]:
            if sorted_scores[ctr, 2] not in training_set.image_superpixels[training_set.all_train_paths[sorted_scores[ctr, 1]]] and not (sorted_scores[ctr, 3] == 1):
                mask = (superpixel_masks[sorted_scores[ctr, 1]] == sorted_scores[ctr, 2]).numpy().astype(np.uint8)
                if training_set.all_train_paths[sorted_scores[ctr, 1]] in selected_regions:  
                    selected_regions[training_set.all_train_paths[sorted_scores[ctr, 1]]] = selected_regions[training_set.all_train_paths[sorted_scores[ctr, 1]]] | mask
                else:
                    selected_regions[training_set.all_train_paths[sorted_scores[ctr, 1]]] = mask
                image_superpixels[training_set.all_train_paths[sorted_scores[ctr, 1]]].append(sorted_scores[ctr, 2])
                valid_pixels = mask.sum()
                total_pixels_selected += valid_pixels
                pbar.update(valid_pixels / (self.base_size[0] * self.base_size[1]))
                if not self.overlap_handler is None:
                    overlapping_indices = []
                    tgt_scene_id = indoor_scenes.IndoorScenesWithAllInfo.get_scene_id_from_image_path(self.dataset, training_set.all_train_paths[sorted_scores[ctr, 1]])
                    overlap_dict = self.overlap_handler.get_overlap_dict_for_scene(tgt_scene_id)
                    tgt_scene_list_index = all_train_scenes.index(tgt_scene_id)
                    sorted_scores_view_mask = sorted_scores[:, 0] == tgt_scene_list_index
                    sorted_scores_view = sorted_scores[sorted_scores_view_mask]

                    for sc_idx in range(sorted_scores_view.shape[0]):
                        src_scene_id = indoor_scenes.IndoorScenesWithAllInfo.get_scene_id_from_image_path(self.dataset, training_set.all_train_paths[sorted_scores_view[sc_idx, 1]])
                        if sorted_scores[ctr, 1] in overlap_dict and (sorted_scores[ctr, 2], sorted_scores_view[sc_idx, 1], sorted_scores_view[sc_idx, 2]) in overlap_dict[sorted_scores[ctr, 1]]:
                            if overlap_dict[sorted_scores[ctr, 1]][(sorted_scores[ctr, 2], sorted_scores_view[sc_idx, 1], sorted_scores_view[sc_idx, 2])] > self.overlap_handler.superpixel_overlap:
                                sorted_scores_view[sc_idx, 3] = 1
                    sorted_scores[sorted_scores_view_mask] = sorted_scores_view

            ctr += 1

        pbar.close()
        print('Selected ', total_pixels_selected / (self.base_size[0] * self.base_size[1]), 'images')
        model.eval()
        training_set.expand_training_set(selected_regions, image_superpixels)

    # window based selection methods

    def nms(self, img_idx, score_map):
        selected_score_map_pts = []
        for i in range((score_map.shape[0]*score_map.shape[1])//(self.region_size*self.region_size)):
            argmax = score_map.view(-1).argmax()
            r, c = argmax // score_map.shape[1], argmax % score_map.shape[1]
            selected_score_map_pts.append((img_idx, r.cpu().item(), c.cpu().item(), score_map[r, c].cpu().item()))
            score_map[max(0, r - self.region_size): min(score_map.shape[0], r + self.region_size), max(0, c - self.region_size): min(score_map.shape[1], c + self.region_size)] = 0

        return selected_score_map_pts

    

    def select_next_batch_with_windows(self, model, training_set, selection_count):
        model.eval()
        model.apply(turn_on_dropout)
        
        weights = torch.cuda.FloatTensor(self.region_size, self.region_size).fill_(1.)
        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, training_set.all_train_paths), batch_size=self.batch_size, shuffle=False, num_workers=0)
        map_ctr = 0
        scores = []
        
        for sample in tqdm(loader, desc='Entropy'):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            for batch_idx, entropy_map in enumerate(self.vote_entropy_selector.batch_entropy_func(model, image_batch, label_batch)):
                if training_set.all_train_paths[map_ctr] in training_set.get_selections():
                    entropy_map[training_set.get_selections()[training_set.all_train_paths[map_ctr]] == 1] = 0
                convolution_output = torch.nn.functional.conv2d(torch.cuda.FloatTensor(entropy_map).unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
                scores.extend(self.nms(map_ctr, convolution_output))
                map_ctr += 1

        selected_samples = sorted(scores, key=lambda x: x[3], reverse=True)[:int(0.5 + selection_count * self.base_size[0] * self.base_size[1] / (self.region_size * self.region_size))]
        print('Last selected sample: ', selected_samples[-1])
        selected_regions = OrderedDict()
        
        total_pixels_selected = 0
        for ss in selected_samples:
            mask = np.zeros(self.base_size, dtype=np.int) == 1
            mask[ss[1] : ss[1] + self.region_size, ss[2] : ss[2] + self.region_size] = True
            valid_pixels = mask.sum()
            total_pixels_selected += valid_pixels    
            if training_set.all_train_paths[ss[0]] in selected_regions:  
                selected_regions[training_set.all_train_paths[ss[0]]] = selected_regions[training_set.all_train_paths[ss[0]]] | mask
            else:
                selected_regions[training_set.all_train_paths[ss[0]]] = mask


        model.eval()
        print('Selected ', total_pixels_selected / (self.base_size[0] * self.base_size[1]), 'images')
        training_set.expand_training_set(selected_regions, [])

