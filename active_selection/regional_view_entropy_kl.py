import numpy as np
import os, sys
from dataloader.indoor_scenes import IndoorScenesWithAllInfo
from utils.misc import visualize_entropy, visualize_spx_dataset
import constants
from dataloader import indoor_scenes
from collections import OrderedDict, defaultdict
from model.deeplab import DeepLab
import torch
from PIL import Image
from active_selection import view_entropy
from dataloader import dataset_base
from utils.saver import Saver
from tqdm import tqdm


class RegionalViewEntropyWithKldivSelector:

    def __init__(self, dataset, lmdb_handle, superpixel_dir, base_size, num_classes, region_size, overlap_handler, mode):
        self.lmdb_handle = lmdb_handle
        self.base_size = base_size
        self.region_size = region_size
        self.dataset = dataset
        self.class_weights = None
        self.overlap_handler = overlap_handler
        self.view_entropy_selector_entropy = view_entropy.ViewEntropySelector(dataset, lmdb_handle, superpixel_dir, base_size, num_classes, 'entropy', mc_dropout=True, superpixel_averaged_maxed=(mode=='superpixel'), return_non_reduced_maps=True)
        self.view_entropy_selector_kl = view_entropy.ViewEntropySelector(dataset, lmdb_handle, superpixel_dir, base_size, num_classes, 'kldiv', mc_dropout=True, superpixel_averaged_maxed=(mode=='superpixel'), return_non_reduced_maps=True)
        if mode == 'window':
            raise NotImplementedError
        elif mode == 'superpixel':
            self.select_next_batch = self.select_next_batch_with_superpixels
        else:
            raise NotImplementedError

    def select_next_batch_with_superpixels(self, model, training_set, selection_count):
        model.eval()
        # get view entropy score
        scores_entropy, image_paths_entropy, _, superpixel_masks_entropy, precomputed_probabilities = self.view_entropy_selector_entropy.calculate_scores(model, training_set.all_train_paths, save_probabilites=False)
        # get view divergence score
        scores_kldiv, image_paths_kldiv, _, superpixel_masks_kldiv, _ = self.view_entropy_selector_kl.calculate_scores(model, training_set.all_train_paths, precomputed_probabilities=precomputed_probabilities)

        for i in range(len(image_paths_entropy)):
            assert image_paths_entropy[i] == image_paths_kldiv[i], "maps not equivalent for kl and entropy"

        del image_paths_kldiv, superpixel_masks_kldiv

        original_image_indices = [training_set.all_train_paths.index(im_path) for im_path in image_paths_entropy]
        all_train_scenes = sorted(list(set([IndoorScenesWithAllInfo.get_scene_id_from_image_path(self.dataset, x) for x in training_set.all_train_paths])))
        original_scene_indices = [all_train_scenes.index(IndoorScenesWithAllInfo.get_scene_id_from_image_path(self.dataset, im_path)) for im_path in image_paths_entropy]

        superpixel_ids = []
        superpixel_scores_expanded = []
        
        # prefetch superpixel maps
        for image_score_idx in range(len(scores_entropy)): 
            superpixel_indices = list(set(scores_entropy[image_score_idx].keys()).intersection(set(scores_kldiv[image_score_idx].keys())))
            for superpixel_idx in superpixel_indices:
                superpixel_ids.append((original_scene_indices[image_score_idx], original_image_indices[image_score_idx], superpixel_idx, image_score_idx, scores_kldiv[image_score_idx][superpixel_idx]))
                superpixel_scores_expanded.append(scores_entropy[image_score_idx][superpixel_idx])

        # sort by view entropy scores
        _sorted_scores = np.array(list(list(zip(*sorted(zip(superpixel_ids, superpixel_scores_expanded), key=lambda x: x[1], reverse=True)))[0]))
        sorted_scores = np.zeros((_sorted_scores.shape[0], _sorted_scores.shape[1] + 1), dtype=np.int32)
        sorted_scores[:, 0:_sorted_scores.shape[1]] = _sorted_scores

        total_pixels_selected = 0
        selected_regions = OrderedDict()
        image_superpixels = defaultdict(list)
        ctr = 0

        print('Selecting superpixels...')
        pbar = tqdm(total=selection_count)

        # while pixels < requested pixels
        while total_pixels_selected < selection_count * self.base_size[0] * self.base_size[1] and ctr < sorted_scores.shape[0]:
            
            # if superpixel is not already labeled or selected 
            if sorted_scores[ctr, 2] not in training_set.image_superpixels[image_paths_entropy[sorted_scores[ctr, 3]]] and not (sorted_scores[ctr, 5] == 1):
                
                tgt_scene_id = IndoorScenesWithAllInfo.get_scene_id_from_image_path(self.dataset, image_paths_entropy[sorted_scores[ctr, 3]])
                overlap_dict = self.overlap_handler.get_overlap_dict_for_scene(tgt_scene_id)
                tgt_scene_list_index = all_train_scenes.index(tgt_scene_id)
                sorted_scores_view_mask = sorted_scores[:, 0] == tgt_scene_list_index
                sorted_scores_view = sorted_scores[sorted_scores_view_mask]

                # get all candidate superpixels based on overlaps across views 
                candidates = []
                for sc_idx in range(sorted_scores_view.shape[0]):
                    if sorted_scores_view[sc_idx, 5] != 1 and sorted_scores_view[sc_idx, 2] not in training_set.image_superpixels[image_paths_entropy[sorted_scores_view[sc_idx, 3]]]:
                        # if ctr intersects sc_idx
                        if sorted_scores[ctr, 1] in overlap_dict and (sorted_scores[ctr, 2], sorted_scores_view[sc_idx, 1], sorted_scores_view[sc_idx, 2]) in overlap_dict[sorted_scores[ctr, 1]]:
                            if overlap_dict[sorted_scores[ctr, 1]][(sorted_scores[ctr, 2], sorted_scores_view[sc_idx, 1], sorted_scores_view[sc_idx, 2])] > self.overlap_handler.superpixel_overlap:
                                candidates.append((sc_idx, sorted_scores_view[sc_idx, 4]))
                        # if sc_idx intersects ctr
                        if sorted_scores_view[sc_idx, 1] in overlap_dict and (sorted_scores_view[sc_idx, 2], sorted_scores[ctr, 1], sorted_scores[ctr, 2]) in overlap_dict[sorted_scores_view[sc_idx, 1]]:
                            if overlap_dict[sorted_scores_view[sc_idx, 1]][(sorted_scores_view[sc_idx, 2], sorted_scores[ctr, 1], sorted_scores[ctr, 2])] > self.overlap_handler.superpixel_overlap:                                
                                candidates.append((sc_idx, sorted_scores_view[sc_idx, 4]))

                winner_img_score_idx, winner_spx_idx = None, None

                # choose superpixel with max view divergence score
                if len(candidates) > 0:
                    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    winner_img_score_idx, winner_spx_idx = sorted_scores_view[candidates[0][0], 3], sorted_scores_view[candidates[0][0], 2]
                    for sc_idx in range(sorted_scores_view.shape[0]):
                        if sorted_scores_view[sc_idx, 5] != 1:
                            # if candidates[0][0] intersects sc_idx
                            if sorted_scores[candidates[0][0], 1] in overlap_dict and (sorted_scores[candidates[0][0], 2], sorted_scores_view[sc_idx, 1], sorted_scores_view[sc_idx, 2]) in overlap_dict[sorted_scores[candidates[0][0], 1]]:
                                if overlap_dict[sorted_scores[candidates[0][0], 1]][(sorted_scores[candidates[0][0], 2], sorted_scores_view[sc_idx, 1], sorted_scores_view[sc_idx, 2])] > self.overlap_handler.superpixel_overlap:
                                    sorted_scores_view[sc_idx, 5] = 1
                            # if sc_idx intersects candidates[0][0]
                            if sorted_scores_view[sc_idx, 1] in overlap_dict and (sorted_scores_view[sc_idx, 2], sorted_scores[candidates[0][0], 1], sorted_scores[candidates[0][0], 2]) in overlap_dict[sorted_scores_view[sc_idx, 1]]:
                                if overlap_dict[sorted_scores_view[sc_idx, 1]][(sorted_scores_view[sc_idx, 2], sorted_scores[candidates[0][0], 1], sorted_scores[candidates[0][0], 2])] > self.overlap_handler.superpixel_overlap:
                                    sorted_scores_view[sc_idx, 5] = 1
                    for sc_idx, _ in candidates:
                        sorted_scores_view[sc_idx, 5] = 1
                else:
                    winner_img_score_idx, winner_spx_idx = sorted_scores[ctr, 3], sorted_scores[ctr, 2]

                sorted_scores[sorted_scores_view_mask] = sorted_scores_view
                
                mask = (superpixel_masks_entropy[winner_img_score_idx] == winner_spx_idx).astype(np.uint8)
                if image_paths_entropy[winner_img_score_idx] in selected_regions:  
                    selected_regions[image_paths_entropy[winner_img_score_idx]] = selected_regions[image_paths_entropy[winner_img_score_idx]] | mask
                else:
                    selected_regions[image_paths_entropy[winner_img_score_idx]] = mask

                image_superpixels[image_paths_entropy[winner_img_score_idx]].append(winner_spx_idx)

                # update progress bar
                valid_pixels = mask.sum()
                total_pixels_selected += valid_pixels
                pbar.update(valid_pixels / (self.base_size[0] * self.base_size[1]))
                
            ctr += 1

        pbar.close()
        print('Selected ', total_pixels_selected / (self.base_size[0] * self.base_size[1]), 'images')
        
        # add to training set (oracle labeling)
        training_set.expand_training_set(selected_regions, image_superpixels)
