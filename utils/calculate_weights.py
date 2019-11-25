import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


def calculate_weights_labels(dataset):

    z = np.zeros((dataset.num_classes,))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Calculating class weights..")

    for sample in tqdm(dataloader):
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = np.logical_and((y >= 0), (y < dataset.num_classes))
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=dataset.num_classes)
        z += count_l

    z = np.nan_to_num(np.sqrt(1 + z))
    total_frequency = np.sum(z)
    class_weights = []

    for frequency in z:
        class_weight = 1 / frequency
        class_weights.append(class_weight)

    ret = np.nan_to_num(np.array(class_weights))
    ret[ret > 2 * np.median(ret)] = 2 * np.median(ret)
    ret = ret / ret.sum()
    print('Class weights: ')
    print(ret)
    return ret
