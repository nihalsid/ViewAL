from dataloader import dataset_base
from dataloader import custom_transforms


class PathsDataset(dataset_base.DatasetBase):

    def __init__(self, lmdb_handle, base_size, paths):
        super(PathsDataset, self).__init__(lmdb_handle, base_size)
        self.image_path_subset = paths

    def __getitem__(self, index):

        image_path = self.image_path_subset[index]
        loaded_npy = self.lmdb_handle.get_numpy_object(image_path)
        image = loaded_npy[:, :, 0:3]
        target = loaded_npy[:, :, 3]

        ret_dict = custom_transforms.transform_validation_sample(image, target)
        return ret_dict
