from dataloader import dataset_base
from pathlib import Path
import constants
from dataloader import custom_transforms
import os
from PIL import Image
import numpy as np
from collections import defaultdict
from numpy.linalg import inv
import math
from skimage.io import imread
from collections import OrderedDict
from tqdm import tqdm

def get_num_classes(dataset):
    if dataset.startswith('scenenet'):
        return 13
    return 40

class IndoorScenes(dataset_base.DatasetBase):

    def __init__(self, dataset, lmdb_handle, base_size, split):

        super(IndoorScenes, self).__init__(lmdb_handle, base_size)
        self.num_classes = get_num_classes(dataset) 
        self.dataset_name = dataset

        with open(os.path.join(constants.SSD_DATASET_ROOT, dataset, "selections", f"{split}_frames.txt"), "r") as fptr:
            self.image_path_subset = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
        
        if split == 'train':
            self.augmentation_transform = custom_transforms.transform_training_sample
        else:
            self.augmentation_transform = custom_transforms.transform_validation_sample

def get_active_dataset(method_name):
    if method_name.endswith("_region"):
        return ActiveIndoorScenesRegional
    elif method_name == 'ceal':
        return ActiveIndoorScenesPseudoLabeled
    else:
        return ActiveIndoorScenes

class ActiveIndoorScenes(IndoorScenes):

    def __init__(self, dataset, lmdb_handle, _, base_size, seed_set):

        super(ActiveIndoorScenes, self).__init__(dataset, lmdb_handle, base_size, seed_set)
        self.remaining_image_paths = None
        all_train_paths = None
        with open(os.path.join(constants.SSD_DATASET_ROOT, dataset, "selections", "train_frames.txt"), "r") as fptr:
            all_train_paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
            self.remaining_image_paths = [x for x in all_train_paths if x not in self.image_path_subset]

    def get_labeled_pixel_count(self):
        return len(self.image_path_subset) * self.base_size[0] * self.base_size[1]

    def get_fraction_of_labeled_data(self):
        return self.get_labeled_pixel_count() / ((len(self.image_path_subset) + len(self.remaining_image_paths)) * self.base_size[0] * self.base_size[1])

    def expand_training_set(self, paths):
        self.image_path_subset.extend(paths)
        for x in paths:
            if x in self.remaining_image_paths:
                self.remaining_image_paths.remove(x)

    def get_selections(self):
        return self.image_path_subset

    def load_selections(self, path):
        self.remaining_image_paths = None
        with open(path, "r") as fptr:
            self.image_path_subset = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
        
        all_train_paths = None
        with open(os.path.join(constants.SSD_DATASET_ROOT, self.dataset_name, "selections", "train_frames.txt"), "r") as fptr:
            all_train_paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
            self.remaining_image_paths = [x for x in all_train_paths if x not in self.image_path_subset]
        
class ActiveIndoorScenesPseudoLabeled(IndoorScenes):

    def __init__(self, dataset, lmdb_handle, _, base_size, seed_set):
        super(ActiveIndoorScenesPseudoLabeled, self).__init__(dataset, lmdb_handle, base_size, seed_set)
        self.path_to_pixel_map = OrderedDict({})
        self.remaining_image_paths = None
        all_train_paths = None
        with open(os.path.join(constants.SSD_DATASET_ROOT, dataset, "selections", "train_frames.txt"), "r") as fptr:
            all_train_paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
            self.remaining_image_paths = [x for x in all_train_paths if x not in self.image_path_subset]
        for path in self.image_path_subset:
            self.path_to_pixel_map[path] = self.lmdb_handle.get_numpy_object(path)[:, :, 3]
        self.image_path_subset_labeled_pseudolabeled = self.image_path_subset[:]

    def __getitem__(self, index):
        image_path = self.image_path_subset_labeled_pseudolabeled[index]
        loaded_npy = self.lmdb_handle.get_numpy_object(image_path)
        image = loaded_npy[:, :, 0:3]
        target = self.path_to_pixel_map[image_path]

        return self.augmentation_transform(image, target, self.base_size)
    
    def get_labeled_pixel_count(self):
        return len(self.image_path_subset) * self.base_size[0] * self.base_size[1]

    def __len__(self):
        return len(self.image_path_subset_labeled_pseudolabeled)

    def get_fraction_of_labeled_data(self):
        return self.get_labeled_pixel_count() / ((len(self.image_path_subset) + len(self.remaining_image_paths)) * self.base_size[0] * self.base_size[1])

    def get_selections(self):
        return self.image_path_subset

    def expand_training_set(self, selected_images_labeled, selected_images_labeled_pseudolabeled):
        self.image_path_subset.extend(selected_images_labeled)
        for x in selected_images_labeled:
            if x in self.remaining_image_paths:
                self.remaining_image_paths.remove(x)
        self.path_to_pixel_map = OrderedDict({})
        for path in self.image_path_subset:
            self.path_to_pixel_map[path] = self.lmdb_handle.get_numpy_object(path)[:, :, 3]

        self.image_path_subset_labeled_pseudolabeled = self.image_path_subset[:]
        
        for path in selected_images_labeled_pseudolabeled:
            if path not in self.path_to_pixel_map:
                self.path_to_pixel_map[path] = selected_images_labeled_pseudolabeled[path]
            self.image_path_subset_labeled_pseudolabeled.append(path)

    def reset_dataset(self):
        self.image_path_subset_labeled_pseudolabeled = self.image_path_subset_labeled_pseudolabeled[:self.len_image_path_subset_labeled_pseudolabeled]

    def make_dataset_multiple_of_batchsize(self, batch_size):
        self.len_image_path_subset_labeled_pseudolabeled = len(self.image_path_subset_labeled_pseudolabeled)
        self.image_path_subset_labeled_pseudolabeled = self._fix_list_multiple_of_batch_size(self.image_path_subset_labeled_pseudolabeled, batch_size)

class ActiveIndoorScenesRegional(IndoorScenes):

    def __init__(self, dataset, lmdb_handle, superpixel_dir, base_size, seed_set):
        super(ActiveIndoorScenesRegional, self).__init__(dataset, lmdb_handle, base_size, seed_set)
        self.all_train_paths = None

        with open(os.path.join(constants.SSD_DATASET_ROOT, dataset, "selections", "train_frames.txt"), "r") as fptr:
            self.all_train_paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
        
        self.path_to_pixel_map = OrderedDict({})

        all_info_dataset = IndoorScenesWithAllInfo(dataset, lmdb_handle, superpixel_dir, base_size, self.image_path_subset)

        self.image_superpixels = defaultdict(list)
        for i in tqdm(range(len(all_info_dataset)), desc='AISR[Unique]'):
            self.image_superpixels[self.image_path_subset[i]] = np.unique(all_info_dataset[i]['superpixel']).tolist()

        for path in self.image_path_subset:
            self.path_to_pixel_map[path] = np.ones(base_size, dtype=np.uint8)


    def __getitem__(self, index):

        image_path = self.image_path_subset[index]
        loaded_npy = self.lmdb_handle.get_numpy_object(image_path)
        mask = self.path_to_pixel_map[image_path]

        image = loaded_npy[:, :, 0:3]
        target = loaded_npy[:, :, 3]
        target[mask == 0] = 255

        return self.augmentation_transform(image, target, self.base_size)

    def get_labeled_pixel_count(self):
        pixel_count = 0
        for path in self.path_to_pixel_map.keys():
            pixel_count += self.path_to_pixel_map[path].sum()
        return pixel_count

    def get_fraction_of_labeled_data(self):
        return self.get_labeled_pixel_count() / (len(self.all_train_paths) * self.base_size[0] * self.base_size[1])

    def expand_training_set(self, region_dict, image_superpixels):
        for image in image_superpixels:
            self.image_superpixels[image].extend(image_superpixels[image])

        for path in region_dict:
            if path in self.path_to_pixel_map:
                self.path_to_pixel_map[path] = self.path_to_pixel_map[path] | region_dict[path]
            else:
                self.path_to_pixel_map[path] = region_dict[path]
                self.image_path_subset.append(path)

    def load_selections(self, path):
        self.path_to_pixel_map = OrderedDict({})
        self.image_path_subset = []
        for p in tqdm(os.listdir(path), desc='load_selections'):
            encoded_p = u'{}'.format(p.split(".")[0]).encode('ascii')
            self.path_to_pixel_map[encoded_p] = imread(os.path.join(path, p)).astype(np.bool)
            self.image_path_subset.append(encoded_p)

    def get_selections(self):
        return self.path_to_pixel_map

class IndoorScenesWithAllInfo(dataset_base.DatasetBase):

    def __init__(self, dataset, lmdb_handle, superpixel_dir, base_size, paths):
        super(IndoorScenesWithAllInfo, self).__init__(lmdb_handle, base_size)
        self.num_classes = get_num_classes(dataset)
        self.dataset = dataset
        self.image_path_subset = paths
        self.scene_id_to_index = defaultdict(list)
        self.superpixel_dir = superpixel_dir
        self.pose_separator = " "
        if dataset.startswith("suncg"):
            self.process_info = self.process_info_suncg
            self.depth_ext = ".png"
            self.scene_id_split_idx = 1
            self.info_id_split_idx = 1
            self.process_superpixels = self._process_superpixels
        elif dataset.startswith('scannet'):
            self.process_info = self.process_info_scannet
            self.depth_ext = ".pgm"
            self.scene_id_split_idx = 2
            self.info_id_split_idx = 2
            self.process_superpixels = self._process_superpixels
        elif dataset.startswith('colmap'):
            self.process_info = self.process_info_colmap
            self.depth_ext = ".png"
            self.pose_separator = ","
            self.scene_id_split_idx = 2
            self.info_id_split_idx = 2
            self.process_superpixels = self._process_superpixels
        elif dataset.startswith('matterport3d'):
            self.process_info = self.process_info_matterport
            self.depth_ext = ".png"
            self.scene_id_split_idx = 2
            self.info_id_split_idx = 3
            self.process_superpixels = self._process_superpixels
        elif dataset.startswith('scenenet-rgbd'):
            self.process_info = self.process_info_scenenet
            self.depth_ext = ".png"
            self.scene_id_split_idx = 2
            self.info_id_split_idx = 1
            self.pose_separator = ","
            self.process_superpixels = self._process_superpixels
        for i, im_path in enumerate(paths):
            scene_id = "_".join(im_path.decode().split("_")[:self.scene_id_split_idx])
            self.scene_id_to_index[scene_id].append(i)

    @staticmethod
    def get_scene_id_from_image_path(dataset, image_path):
        if dataset.startswith("suncg"):
            scene_id_split_idx = 1
        elif dataset.startswith('scannet'):
            scene_id_split_idx = 2
        elif dataset.startswith('colmap'):
            scene_id_split_idx = 2
        elif dataset.startswith('matterport3d'):
            scene_id_split_idx = 2
        elif dataset.startswith('scenenet-rgbd'):
            scene_id_split_idx = 2
        return "_".join(image_path.decode().split("_")[:scene_id_split_idx])

    def process_depth(self, depth_path):
        return np.asarray(Image.open(depth_path).resize((constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT))) / 1000.0

    def process_info_scannet(self, info_path):
        matrix_members = None
        with open(info_path, "r") as fptr:
            matrix_members = [float(x.strip()) for x in fptr.readlines()[9].split("=")[1].split(" ") if x.strip() != ""]
        matrix = np.array(matrix_members).reshape(4, 4)
        matrix[0, 0] /= 2
        matrix[1, 1] /= 2
        matrix[0, 2] /= 2
        matrix[1, 2] /= 2
        return matrix
    
    def process_info_colmap(self, info_path):
        matrix = np.zeros((4, 4))
        with open(info_path, "r") as fptr:
            lines = [x.strip() for x in fptr.readlines() if x.strip()!=""]
            for r in range(len(lines)):
                elements = [float(x.strip()) for x in lines[r].split(",") if x.strip()!=""]
                for c in range(len(elements)):
                    matrix[r, c] = elements[c]
            matrix[3, 3] = 1
        return matrix 


    def process_info_matterport(self, info_path):
        info_path_as_path = Path(info_path)
        fixed_file_name = info_path_as_path.name.split("_")[0]+"_"+info_path_as_path.name.split("_")[2]
        fixed_path = str(info_path_as_path.parent / fixed_file_name)
        matrix_members = None
        with open(fixed_path, "r") as fptr:
            matrix_members = [float(x.strip()) for x in fptr.readlines()[9].split("=")[1].split(" ") if x.strip() != ""]
        matrix = np.array(matrix_members).reshape(4, 4)
        matrix[0, 0] /= 4
        matrix[1, 1] /= 4.2666667
        matrix[0, 2] /= 4
        matrix[1, 2] /= 4.2666667
        return matrix

    def process_info_scenenet(self, _):
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = 277.1281292110204
        matrix[1, 1] = 289.7056274847714
        matrix[0, 2] = 160
        matrix[1, 2] = 120
        matrix[2, 2] = 1
        matrix[0, 3] = 0
        matrix[1, 3] = 0
        matrix[2, 3] = 0
        matrix[3, 3] = 1
        return matrix

    def process_info_suncg(self, _):
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = 160 / math.tan(math.radians(30))
        matrix[1, 1] = 120 / math.tan(math.radians(30))
        matrix[0, 2] = 160
        matrix[1, 2] = 120
        matrix[2, 2] = 1
        matrix[0, 3] = 0
        matrix[1, 3] = 0
        matrix[2, 3] = 0
        matrix[3, 3] = 1
        return matrix

    def process_pose(self, pose_path):
        matrix_members = []
        with open(pose_path, "r") as fptr:
            for line in fptr.readlines():
                matrix_members.extend([float(x.strip()) for x in line.split(self.pose_separator) if x.strip() != ""])
        return np.array(matrix_members).reshape(4, 4)

    def _process_superpixels(self, superpixel_path):
        return np.asarray(imread(superpixel_path), dtype=np.int32)

    def _process_superpixels_dummy(self, superpixel_path):
        return np.zeros((240, 320), dtype=np.uint16)

    def __getitem__(self, index):

        image_path = self.image_path_subset[index]
        loaded_npy = self.lmdb_handle.get_numpy_object(image_path)

        image_path = image_path.decode()
        scene_id = "_".join(image_path.split("_")[:self.scene_id_split_idx])
        info_id = "_".join(image_path.split("_")[:self.info_id_split_idx])
        depth_image_path = os.path.join(constants.SSD_DATASET_ROOT, self.dataset, "raw", "selections", "depth", f"{image_path}{self.depth_ext}")
        info_path = os.path.join(constants.SSD_DATASET_ROOT, self.dataset, "raw", "selections", "info", f"{info_id}.txt")
        pose_path = os.path.join(constants.SSD_DATASET_ROOT, self.dataset, "raw", "selections", "pose", f"{image_path}.txt")
        spx_path = os.path.join(constants.SSD_DATASET_ROOT, self.dataset, "raw", "selections", self.superpixel_dir, f"{image_path}.png")

        image = loaded_npy[:, :, 0:3]
        target = loaded_npy[:, :, 3]
        depth_image = self.process_depth(depth_image_path)
        depth_intrinsic = self.process_info(info_path)
        pose = self.process_pose(pose_path)
        spx = self.process_superpixels(spx_path)

        ret_dict = custom_transforms.transform_validation_sample(image, target)
        ret_dict['depth'] = depth_image
        ret_dict['intrinsic'] = depth_intrinsic
        ret_dict['pose'] = pose
        ret_dict['scene_id'] = scene_id
        ret_dict['superpixel'] = spx

        return ret_dict
