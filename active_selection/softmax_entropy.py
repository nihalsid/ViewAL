from dataloader.paths import PathsDataset
from utils.misc import turn_on_dropout, visualize_entropy
import constants
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class SoftmaxEntropySelector:

    def __init__(self, dataset, lmdb_handle, base_size, batch_size, num_classes):
        self.lmdb_handle = lmdb_handle
        self.num_classes = num_classes
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
                num_classes = output.shape[1]
                for batch_idx in range(output.shape[0]):
                    entropy_map = torch.cuda.FloatTensor(output.shape[2], output.shape[3]).fill_(0)
                    for c in range(self.num_classes):
                        entropy_map = entropy_map - (output[batch_idx, c, :, :] * torch.log2(output[batch_idx, c, :, :] + 1e-12))
                    entropy_map[label_batch[batch_idx, :, :] == 255] = 0
                    scores.append(entropy_map.mean().cpu().item())
                    del entropy_map
                torch.cuda.empty_cache()
        return scores


    def select_next_batch(self, model, training_set, selection_count):

        scores = self.calculate_scores(model, training_set.remaining_image_paths)
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        model.eval()
        training_set.expand_training_set(selected_samples)
