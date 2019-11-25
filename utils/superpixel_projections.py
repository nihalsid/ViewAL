import numpy as np
import os
import constants
from numpy.linalg import inv
from dataloader import indoor_scenes
import torch
from collections import OrderedDict, Counter
from tqdm import tqdm

def project_image_to_world(x, y, depth, cam2world, depth_intrinsic):
    I = torch.zeros(4, depth.shape[0]).type(torch.cuda.FloatTensor)
    I[0, :] = x * depth
    I[1, :] = y * depth
    I[2, :] = depth
    I[3, :] = 1.0
    world_coordinates = torch.mm(torch.from_numpy(cam2world).type(torch.cuda.FloatTensor), torch.mm(
        torch.from_numpy(inv(depth_intrinsic)).type(torch.cuda.FloatTensor), I))

    del I, x, y, depth
    torch.cuda.empty_cache()

    return world_coordinates

def project_images_to_world(depths, cam2worlds, depth_intrinsic, superpixels, frames):
    x = np.linspace(0, constants.DEPTH_WIDTH - 1, constants.DEPTH_WIDTH)
    y = np.linspace(0, constants.DEPTH_HEIGHT - 1, constants.DEPTH_HEIGHT)
    x_mesh, y_mesh = np.meshgrid(x, y)

    world_coordinates = torch.zeros(4, len(depths) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT).type(torch.cuda.FloatTensor)
    frame_origins = torch.zeros(len(depths) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT).type(torch.cuda.IntTensor)
    superpixel_origins = torch.zeros(len(depths) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT).type(torch.cuda.IntTensor)

    for im_idx in range(len(depths)):
        world_coordinates[:, im_idx * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT: (im_idx + 1) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT] = project_image_to_world(torch.from_numpy(x_mesh).type(torch.cuda.FloatTensor).flatten(),
                                                                                                                                                                              torch.from_numpy(y_mesh).type(torch.cuda.FloatTensor).flatten(), torch.from_numpy(depths[im_idx][:]).type(torch.cuda.FloatTensor).flatten(), cam2worlds[im_idx], depth_intrinsic)
        frame_origins[im_idx * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT: (im_idx + 1) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT] = torch.ones(
            constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT).type(torch.cuda.IntTensor) * frames[im_idx]

        superpixel_origins[im_idx * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT: (im_idx + 1) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT] = torch.from_numpy(superpixels[im_idx].astype(np.int).flatten()).type(torch.cuda.IntTensor)

    # visualize_point_cloud(world_coordinates)

    return world_coordinates, frame_origins, superpixel_origins

def project_world_to_image(depth, superpixel_map, cam2world, depth_intrinsic, world_coordinates, frame_origins, superpixel_origins):
    world_coordinates_copy = world_coordinates.transpose(0, 1)[:, :3]
    projected_points = torch.mm(torch.mm(torch.from_numpy(depth_intrinsic).type(torch.cuda.FloatTensor),
                                         torch.from_numpy(inv(cam2world)).type(torch.cuda.FloatTensor)), world_coordinates)
    projected_points = projected_points.transpose(0, 1)[:, :3]
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]
    projected_points[:, 2] /= projected_points[:, 2]
    selection_mask = ~torch.isnan(projected_points[:, 2])

    projected_points = torch.round(projected_points[selection_mask])
    frame_origins = frame_origins[selection_mask]
    superpixel_origins = superpixel_origins[selection_mask]
    world_coordinates_copy = world_coordinates_copy[selection_mask]

    # remove out of frame bounds
    selection_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < constants.DEPTH_WIDTH) & (
        projected_points[:, 1] >= 0) & (projected_points[:, 1] < constants.DEPTH_HEIGHT)

    projected_points = projected_points[selection_mask][:, :2]
    frame_origins = frame_origins[selection_mask]
    superpixel_origins = superpixel_origins[selection_mask]
    world_coordinates_copy = world_coordinates_copy[selection_mask]

    depth = torch.from_numpy(depth).type(torch.cuda.FloatTensor)
    depth = depth[projected_points[:, 1].type(torch.cuda.LongTensor), projected_points[:, 0].type(torch.cuda.LongTensor)].flatten()
    backprojected_points = project_image_to_world(projected_points[:, 0], projected_points[
        :, 1], depth, cam2world, depth_intrinsic).transpose(0, 1)[:, :3]

    selection_mask = (torch.norm(world_coordinates_copy - backprojected_points, dim=1) < constants.WORLD_DISTANCE_THRESHOLD)

    projected_points = projected_points[selection_mask]

    if projected_points.shape[0] == 0:
        return None

    frame_origins = frame_origins[selection_mask]
    superpixel_origins = superpixel_origins[selection_mask]
    superpixel_targets = superpixel_map[projected_points[:, 1].type(torch.cuda.LongTensor).cpu().numpy(), projected_points[:, 0].type(torch.cuda.LongTensor).cpu().numpy()].flatten()    
    t1, t2 = np.unique(superpixel_map, return_counts=True)
    target_superpixel_sizes = dict(zip(t1, t2))

    frame_spx = torch.zeros((frame_origins.shape[0], 3)).type(torch.cuda.IntTensor)
    frame_spx[:, 0] = torch.from_numpy(superpixel_targets.astype(np.int)).type(torch.cuda.IntTensor)
    frame_spx[:, 1] = frame_origins
    frame_spx[:, 2] = superpixel_origins

    uniques, counts = torch.unique(frame_spx, dim=0, return_counts=True)
    frame_spx_counts = {}
    for idx, u in enumerate(uniques.tolist()):
        frame_spx_counts[tuple(u)] = float(counts[idx].cpu().item())

    coverage_dict = {}
    for i in frame_spx_counts:
        coverage = frame_spx_counts[i] / target_superpixel_sizes[i[0]]
        coverage_dict[(i[0], i[1], i[2])] = coverage

    return coverage_dict  # , projected_points


def find_superpixel_coverage(dataset_name, lmdb_handle, superpixel_dir, base_size, images):
    dataset = indoor_scenes.IndoorScenesWithAllInfo(dataset_name, lmdb_handle, superpixel_dir, base_size, images)
    scene_id_to_index = dataset.scene_id_to_index
    
    image_paths = []

    for scene_id in tqdm(scene_id_to_index, desc='Scene[Coverage]'):
        all_frame_coverages = OrderedDict()
        depths = []
        poses = []
        superpixels = []
        intrinsic = None    

        for frame_id in scene_id_to_index[scene_id]:
            sample = dataset[frame_id]
            depths.append(sample['depth'])
            poses.append(sample['pose'])
            superpixels.append(sample['superpixel'])
            intrinsic = sample['intrinsic']

        world_coordinates, frame_origins, superpixel_origins = project_images_to_world(depths, poses, intrinsic, superpixels, scene_id_to_index[scene_id])

        for frame_id in tqdm(scene_id_to_index[scene_id], desc='Scene[Project]'):
            sample = dataset[frame_id]
            frame_coverages = project_world_to_image(sample['depth'], sample['superpixel'], sample['pose'], sample['intrinsic'], world_coordinates, frame_origins, superpixel_origins)
            if not frame_coverages is None:
                all_frame_coverages[frame_id] = frame_coverages
                image_paths.append(images[frame_id])
        #from pprint import pprint
        #pprint(all_frame_coverages)
        np.save(os.path.join(constants.SSD_DATASET_ROOT, dataset_name, "raw", "selections", "coverage_"+superpixel_dir, f'{scene_id}.npy'), all_frame_coverages)

        del world_coordinates, frame_origins, superpixel_origins
        del depths, poses, superpixels, all_frame_coverages
        torch.cuda.empty_cache()  

    with open(os.path.join(constants.SSD_DATASET_ROOT, dataset_name, "raw", "selections", "coverage_"+superpixel_dir, "coverage_paths.txt"), "w") as fptr:
        for p in image_paths:
            fptr.write(p.decode() + "\n")
    

def test_coverage_scannet_sample():
    import constants
    import os
    from dataloader import dataset_base
    from dataloader.indoor_scenes import IndoorScenes
    lmdb_handle = dataset_base.LMDBHandle(os.path.join(constants.HDD_DATASET_ROOT, "scannet-sample", "dataset.lmdb"), False)    
    train_set = IndoorScenes('scannet-sample', lmdb_handle, (240, 320), 'train')
    find_superpixel_coverage('scannet-sample', lmdb_handle, (240, 320), train_set.image_path_subset)

if __name__=='__main__':
    test_coverage_scannet_sample()
