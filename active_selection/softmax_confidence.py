from dataloader.paths import PathsDataset
from utils.misc import turn_on_dropout, visualize_entropy
import constants
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class SoftmaxConfidenceSelector:

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
                max_conf_batch = torch.max(softmax(model(image_batch)), dim=1)[0]
                
                for batch_idx in range(max_conf_batch.shape[0]):
                    max_conf_batch[batch_idx, (label_batch[batch_idx, :, :] == 255)] = 1
                    scores.append(torch.mean(max_conf_batch[batch_idx, :, :]).cpu().item())
                del max_conf_batch
                torch.cuda.empty_cache()
        return scores


    def select_next_batch(self, model, training_set, selection_count):

        scores = self.calculate_scores(model, training_set.remaining_image_paths)
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=False)))[1][:selection_count]

        model.eval()
        training_set.expand_training_set(selected_samples)
