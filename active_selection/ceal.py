from dataloader.paths import PathsDataset
from dataloader.indoor_scenes import IndoorScenesWithAllInfo
from utils.misc import visualize_entropy, visualize_spx_dataset
from dataloader import indoor_scenes, dataset_base
import constants
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class CEALSelector:

    def __init__(self, dataset, lmdb_handle, base_size, batch_size, num_classes, start_entropy_threshold, entropy_change_per_selection):
        self.lmdb_handle = lmdb_handle
        self.num_classes = num_classes
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.current_entropy_threshold = start_entropy_threshold
        self.entropy_change_per_selection = entropy_change_per_selection

    def calculate_scores(self, model, paths):
        model.eval()
        scores = []

        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, paths), batch_size=self.batch_size, shuffle=False, num_workers=0)
        path_ctr = 0
        entropy_maps = {}
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()

                softmax = torch.nn.Softmax2d()
                output = softmax(model(image_batch))
                num_classes = output.shape[1]
                for batch_idx in range(output.shape[0]):
                    entropy_map = torch.cuda.FloatTensor(output.shape[2], output.shape[3]).fill_(0)
                    for c in range(self.num_classes):
                        entropy_map = entropy_map - (output[batch_idx, c, :, :] * torch.log2(output[batch_idx, c, :, :] + 1e-12))
                    entropy_map[label_batch[batch_idx, :, :] == 255] = 0
                    scores.append(entropy_map.mean().cpu().item())
                    entropy_maps[paths[path_ctr]] = entropy_map.cpu().numpy()
                    path_ctr += 1
                torch.cuda.empty_cache()
        return scores, entropy_maps


    def select_next_batch(self, model, training_set, selection_count):
        scores, entropy_maps = self.calculate_scores(model, training_set.remaining_image_paths)
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        print(f'Selected Samples: {len(selected_samples)}/{len(training_set.remaining_image_paths)}')
        unselected_samples = [x for x in entropy_maps if x not in selected_samples]

        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, unselected_samples), batch_size=self.batch_size, shuffle=False, num_workers=0)
        pseudolabels = {}
        path_ctr = 0
        pseudo_images_selected = 0
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].numpy()
                output = model(image_batch)
                for batch_idx in range(output.shape[0]):
                    prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0).astype(np.uint8)
                    prediction[label_batch[batch_idx, :, :] == 255] = 255
                    qualified_area = entropy_maps[unselected_samples[path_ctr]] <= self.current_entropy_threshold
                    qualified_area[label_batch[batch_idx, :, :] == 255] = False
                    if np.any(qualified_area):
                        prediction[qualified_area == False] = 255
                        pseudolabels[unselected_samples[path_ctr]] = prediction
                        pseudo_images_selected += qualified_area.sum() / (self.base_size[0]*self.base_size[1])
                    path_ctr += 1
        model.eval()
        print(f'Pseudo Samples: {pseudo_images_selected}/{len(unselected_samples)}')
        training_set.expand_training_set(selected_samples, pseudolabels)
        self.current_entropy_threshold -= self.entropy_change_per_selection

