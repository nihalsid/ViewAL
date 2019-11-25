import lmdb
import os
from torch.utils import data
import PIL
import pickle
import numpy as np
from dataloader import custom_transforms
from tqdm import tqdm


class LMDBHandle:

    def __init__(self, lmdb_path, memory_hog_mode=False):

        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        self.memory_hog_mode = memory_hog_mode

        with self.env.begin(write=False) as txn:
            self.image_paths = pickle.loads(txn.get(b'__keys__'))

        if self.memory_hog_mode:
            self.path_to_npy = {}
            print('Acquiring dataset in memory')
            for n in tqdm(self.image_paths):
                with self.env.begin(write=False) as txn:
                    loaded_npy = pickle.loads(txn.get(n))
                    self.path_to_npy[n] = loaded_npy

    def get_image_paths(self):
        return self.image_paths

    def get_numpy_object(self, image_path):
        loaded_npy = None
        if self.memory_hog_mode and image_path in self.path_to_npy:
            loaded_npy = self.path_to_npy[image_path]
        else:
            with self.env.begin(write=False) as txn:
                loaded_npy = pickle.loads(txn.get(image_path))
        return loaded_npy

class OverlapHandler:

    def __init__(self, overlap_folder, superpixel_overlap, memory_hog_mode, list_of_scenes=None):

        self.memory_hog_mode = memory_hog_mode
        self.scene_overlaps = {}
        self.superpixel_overlap = superpixel_overlap
        self.overlap_folder = overlap_folder
        if memory_hog_mode:
            print('Acquiring overlaps..')
            for f in tqdm(os.listdir(overlap_folder)):
                if f.endswith(".npy"):
                    if not list_of_scenes:
                        self.scene_overlaps[f.split(".")[0]] = np.load(os.path.join(overlap_folder, f), allow_pickle=True)[()]
                    elif f.split(".")[0] in list_of_scenes:
                        self.scene_overlaps[f.split(".")[0]] = np.load(os.path.join(overlap_folder, f), allow_pickle=True)[()]

    def get_overlap_dict_for_scene(self, scene_id):
        loaded_dict = {}
        if self.memory_hog_mode and scene_id in self.scene_overlaps:
            loaded_dict = self.scene_overlaps[scene_id]
        else:
            loaded_dict = np.load(os.path.join(self.overlap_folder, scene_id+".npy"))[()]
        return loaded_dict


class DatasetBase(data.Dataset):

    def __init__(self, lmdb_handle, base_size):

        self.lmdb_handle = lmdb_handle
        self.image_paths = lmdb_handle.get_image_paths()
        self.base_size = base_size
        self.image_path_subset = self.image_paths
        self.augmentation_transform = custom_transforms.transform_training_sample

        if len(self.image_path_subset) == 0:
            raise Exception("No images found in dataset directory")

    def __len__(self):
        return len(self.image_path_subset)

    def __getitem__(self, index):

        image_path = self.image_path_subset[index]
        loaded_npy = self.lmdb_handle.get_numpy_object(image_path)

        image = loaded_npy[:, :, 0:3]
        target = loaded_npy[:, :, 3]

        return self.augmentation_transform(image, target, self.base_size)

    def _fix_list_multiple_of_batch_size(self, paths, batch_size):
        remainder = len(paths) % batch_size
        if remainder != 0:
            num_new_entries = batch_size - remainder
            new_entries = paths[:num_new_entries]
            paths.extend(new_entries)
        return paths

    def reset_dataset(self):
        self.image_path_subset = self.image_path_subset[:self.len_image_path_subset]

    def make_dataset_multiple_of_batchsize(self, batch_size):
        self.len_image_path_subset = len(self.image_path_subset)
        self.image_path_subset = self._fix_list_multiple_of_batch_size(self.image_path_subset, batch_size)

