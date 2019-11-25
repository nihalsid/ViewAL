import numpy as np
from utils.colormaps import map_segmentation_to_colors
import constants
import os
import torch


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def visualize_point_cloud(world_coordinates):
    import open3d as o3d
    xyz = np.transpose(world_coordinates)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


def turn_on_dropout(model):
    if type(model) == torch.nn.Dropout2d:
        model.train()


def visualize_entropy(image_normalized, entropy_map, prediction=None, ground_truth=None, valid_mask=None, weight_map=None, save=False, suffix='ent'):
    import matplotlib
    import matplotlib.pyplot as plt
    from imageio import imwrite
    if not image_normalized is None:
        image_unnormalized = ((np.transpose(image_normalized, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(entropy_map), clip=False)
    norm_ent = matplotlib.colors.Normalize(vmin=0, vmax=visualize_entropy.max_entropy, clip=False)
    norm_weight = matplotlib.colors.Normalize(vmin=0, vmax=visualize_entropy.max_weight, clip=False)
    plt.figure()
    num_subplots = 2
    if not prediction is None:
        num_subplots += 1
    if not ground_truth is None:
        num_subplots += 1
    if not valid_mask is None:
        num_subplots += 1
    if not weight_map is None:
        num_subplots += 1
    cur_subplot = 1
    plt.title('display')
    if not image_normalized is None:
        plt.subplot(1, num_subplots, cur_subplot)
        plt.imshow(image_unnormalized)
        imwrite(os.path.join(constants.RUNS, 'image_dumps', f'img_{visualize_entropy.save_idx:04d}.jpg'), image_unnormalized)
        cur_subplot += 1
    plt.subplot(1, num_subplots, cur_subplot)
    cm_hot = matplotlib.cm.get_cmap('jet')
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'emap_{suffix}_{visualize_entropy.save_idx:04d}.png'), cm_hot(entropy_map / visualize_entropy.max_entropy))
    plt.imshow(entropy_map, norm=norm_ent, cmap='jet')
    cur_subplot += 1
    if not prediction is None:
        prediction_mapped = map_segmentation_to_colors(prediction.astype(np.uint8), 'scannet')
        #imwrite(os.path.join(constants.RUNS, 'image_dumps', f'pred_0_{visualize_entropy.save_idx:04d}.png'), cm_hot(prediction))
        imwrite(os.path.join(constants.RUNS, 'image_dumps', f'pred_{visualize_entropy.save_idx:04d}.png'), prediction_mapped)
        plt.subplot(1, num_subplots, cur_subplot)
        cur_subplot += 1
        plt.imshow(prediction_mapped)
    if not ground_truth is None:
        ground_truth = map_segmentation_to_colors(ground_truth.astype(np.uint8), 'scannet')
        #imwrite(os.path.join(constants.RUNS, 'image_dumps', f'pred_0_{visualize_entropy.save_idx:04d}.png'), cm_hot(prediction))
        imwrite(os.path.join(constants.RUNS, 'image_dumps', f'gt_{visualize_entropy.save_idx:04d}.png'), ground_truth)
        plt.subplot(1, num_subplots, cur_subplot)
        cur_subplot += 1
        plt.imshow(ground_truth)
    if not valid_mask is None:
        plt.subplot(1, num_subplots, cur_subplot)
        cur_subplot += 1
        plt.imshow(valid_mask)
    if not weight_map is None:
        plt.subplot(1, num_subplots, cur_subplot)
        cur_subplot += 1
        plt.imshow(weight_map, norm=norm_weight, cmap='jet')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(constants.RUNS, 'image_dumps', f'{visualize_entropy.save_idx:04d}_{suffix}.jpg'), bbox_inches='tight')
        visualize_entropy.save_idx += 1
        plt.close()
    else:
        plt.show(block=not save)

visualize_entropy.save_idx = 0
visualize_entropy.max_weight = 1
visualize_entropy.max_entropy = np.log2(40 / 2)


def visualize_vote_view_entropy(lmdb_handle, base_size, paths, indices_to_dataset, vote_entropy_scores, view_entropy_scores, scores):
    from dataloader.paths import PathsDataset
    dataset = PathsDataset(lmdb_handle, base_size, paths)

    for i, j in zip(indices_to_dataset, range(len(indices_to_dataset))):
        import matplotlib
        import matplotlib.pyplot as plt
        image_unnormalized = ((np.transpose(dataset[i]['image'].numpy(), axes=[1, 2, 0]) *
                               (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)

        plt.figure()
        plt.title('display')
        plt.subplot(1, 4, 1)
        plt.imshow(image_unnormalized)
        plt.subplot(1, 4, 2)
        norm_ent = matplotlib.colors.Normalize(vmin=0, vmax=visualize_entropy.max_entropy, clip=False)
        plt.imshow(vote_entropy_scores[j, :, :], norm=norm_ent, cmap='jet')
        plt.subplot(1, 4, 3)
        plt.imshow(view_entropy_scores[j, :, :], norm=norm_ent, cmap='jet')
        plt.subplot(1, 4, 4)
        norm = matplotlib.colors.Normalize(vmin=np.min(scores[j]), vmax=np.max(scores[j]), clip=False)
        plt.imshow(scores[j], norm=norm, cmap='jet')
        plt.savefig(os.path.join(constants.RUNS, 'image_dumps', f'ent_{visualize_entropy.save_idx:04d}.png'), bbox_inches='tight')
        visualize_entropy.save_idx += 1
        plt.close()
        plt.show(block=False)


def mark_boundaries(image_0, image_1, color):
    from scipy import ndimage
    boundary_mask = np.ones_like(image_1)
    for i in range(image_1.shape[0] - 1):
        for j in range(image_1.shape[1] - 1):
            if (image_1[i, j] != image_1[i, j + 1] or image_1[i, j] != image_1[i + 1, j]):
                boundary_mask[i, j] = 0
    boundary_mask = ndimage.binary_erosion(boundary_mask, structure=np.ones((2, 2))).astype(boundary_mask.dtype)
    image_0[boundary_mask == 0, :] = color
    return image_0


def visualize_spx_dataset(dataset_name, dataset):
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from dataloader.indoor_scenes import IndoorScenesWithAllInfo
    
    dataset.image_path_subset = sorted(dataset.image_path_subset)
    spx_dataset = IndoorScenesWithAllInfo(dataset_name, dataset.lmdb_handle, "superpixel", (240, 320), dataset.image_path_subset)

    for i in tqdm(range(len(dataset)), desc='Visualization'):
        plt.figure(figsize=(16,8))
        plt.title('spx')
        plt.subplot(1, 2, 1)
        image_unnormalized = ((np.transpose(dataset[i]['image'].numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255)
        image_unnormalized[dataset[i]['label'].numpy()==255, :] *= 0.5
        image_unnormalized = image_unnormalized.astype(np.uint8)
        image_unnormalized = mark_boundaries(image_unnormalized, spx_dataset[spx_dataset.image_path_subset.index(dataset.image_path_subset[i])]['superpixel'])
        plt.imshow(image_unnormalized)
        plt.subplot(1, 2, 2)
        prediction_mapped = map_segmentation_to_colors(dataset[i]['label'].numpy().astype(np.uint8), dataset_name)
        prediction_mapped = mark_boundaries(prediction_mapped, spx_dataset[spx_dataset.image_path_subset.index(dataset.image_path_subset[i])]['superpixel'])
        plt.imshow(prediction_mapped)
        plt.savefig(os.path.join(constants.RUNS, 'image_dumps', f'sel_{i:04d}.png'), bbox_inches='tight')
        plt.close()

def visualize_numbered_superpixels(dataset_name, dataset):
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from dataloader.indoor_scenes import IndoorScenesWithAllInfo
    
    spx_dataset = IndoorScenesWithAllInfo(dataset_name, dataset.lmdb_handle, "superpixel_40", (240, 320), dataset.image_path_subset)

    for i in tqdm(range(len(dataset)), desc='Visualization'):
        for j in np.unique(spx_dataset[spx_dataset.image_path_subset.index(dataset.image_path_subset[i])]['superpixel']).tolist():
            plt.figure(figsize=(16,8))
            plt.title('spx')
            plt.subplot(1, 2, 1)
            image_unnormalized = ((np.transpose(dataset[i]['image'].numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            mask = spx_dataset[spx_dataset.image_path_subset.index(dataset.image_path_subset[i])]['superpixel']!=j 
            image_unnormalized = mark_boundaries(image_unnormalized, spx_dataset[spx_dataset.image_path_subset.index(dataset.image_path_subset[i])]['superpixel'], [255,255,255])
            image_unnormalized[mask, :] = image_unnormalized[mask, :] // 2
            plt.imshow(image_unnormalized)
            plt.subplot(1, 2, 2)
            prediction_mapped = map_segmentation_to_colors(dataset[i]['label'].numpy().astype(np.uint8), dataset_name)
            prediction_mapped = mark_boundaries(prediction_mapped, spx_dataset[spx_dataset.image_path_subset.index(dataset.image_path_subset[i])]['superpixel'], [255,255,255])
            prediction_mapped[mask, :] = prediction_mapped[mask, :] // 3 
            plt.imshow(prediction_mapped)
            plt.savefig(os.path.join(constants.RUNS, 'image_dumps', f'{i:03d}_{j:03d}.png'), bbox_inches='tight')
            plt.close()    

def visualize_seedset_spx(dataset_name):
    from dataloader import dataset_base
    from dataloader.indoor_scenes import IndoorScenes
    from torch.utils.data import DataLoader
    
    lmdb_handle = dataset_base.LMDBHandle(os.path.join(constants.HDD_DATASET_ROOT, dataset_name, "dataset.lmdb"), False)
    dataset = IndoorScenes(dataset_name, lmdb_handle, (240, 320), 'seedset_0')
    paths = [f'scene0014_00_{i:06d}' for i in [1540]]
    print(paths)
    images = [u'{}'.format(x).encode('ascii') for x in paths]
    dataset.image_path_subset = images
    #visualize_spx_dataset(dataset_name, dataset)
    visualize_numbered_superpixels(dataset_name, dataset)
    
def visualize_selection_spx(dataset_name, selections_path):
    from dataloader import dataset_base
    from dataloader.indoor_scenes import IndoorScenes, ActiveIndoorScenesRegional, IndoorScenesWithAllInfo
    from torch.utils.data import DataLoader

    lmdb_handle = dataset_base.LMDBHandle(os.path.join(constants.HDD_DATASET_ROOT, dataset_name, "dataset.lmdb"), False)
    train_set = ActiveIndoorScenesRegional(dataset_name, lmdb_handle, (240, 320), 'seedset_0')    
    train_set.load_selections(os.path.join(constants.RUNS, dataset_name, selections_path, "selections"))
    visualize_spx_dataset(dataset_name, train_set)

def _mark_boundaries(mask, output):
    from scipy import ndimage
    boundary_mask = np.ones_like(mask)
    for i in range(mask.shape[0] - 1):
        for j in range(mask.shape[1] - 1):
            if (mask[i, j] != mask[i, j + 1] or mask[i, j] != mask[i + 1, j]):
                boundary_mask[i, j] = 0
                #output[i, j, :] = [0, 0, 0]
    boundary_mask = ndimage.binary_erosion(boundary_mask, structure=np.ones((3, 3))).astype(boundary_mask.dtype)
    output[boundary_mask == 0, :] = [0, 0, 0] 
    return output

def visualize_image_target_prediction(filename, image, target, random, random_loss, viewal, viewal_loss, full, full_loss):
    from imageio import imwrite
    import matplotlib
    from PIL import Image

    cm_hot = matplotlib.cm.get_cmap('jet')
    image_unnormalized = ((np.transpose(image, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255)
    c0 = 0.60
    c1 = 1 - c0
    target_mapped = map_segmentation_to_colors(target.astype(np.uint8), 'scannet') * 255
    target_mapped = _mark_boundaries(target, (c0 * target_mapped + c1 * image_unnormalized).astype(np.uint8))
    random_mapped = map_segmentation_to_colors(random.astype(np.uint8), 'scannet') * 255
    random_mapped = _mark_boundaries(random,(c0 * random_mapped + c1 * image_unnormalized).astype(np.uint8))
    viewal_mapped = map_segmentation_to_colors(viewal.astype(np.uint8), 'scannet') * 255
    viewal_mapped = _mark_boundaries(viewal,(c0 * viewal_mapped + c1 * image_unnormalized).astype(np.uint8))
    full_mapped = map_segmentation_to_colors(full.astype(np.uint8), 'scannet') * 255
    full_mapped = _mark_boundaries(full,(c0 * full_mapped + c1 * image_unnormalized).astype(np.uint8))
    
    #hstacked = np.hstack((target_mapped, full_mapped, random_mapped, viewal_mapped))
    normalizer = max(max(np.max(random_loss), np.max(viewal_loss)), np.max(full_loss)) / 1.5
    
    #imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}.png'), hstacked)
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_im.png'), image_unnormalized)
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_tgt.png'), target_mapped)
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_rnd.png'), random_mapped)
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_rnd_loss.png'), cm_hot(random_loss/normalizer))
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_val.png'), viewal_mapped)
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_val_loss.png'), cm_hot(viewal_loss/normalizer))
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_ful.png'), full_mapped)
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_ful_loss.png'), cm_hot(full_loss/normalizer))

    im_im = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_im.png'))
    im_tgt = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_tgt.png'))
    im_rnd = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_rnd.png'))
    im_val = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_val.png'))
    im_ful = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_ful.png'))
    im_rnd_loss = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_rnd_loss.png'))
    im_val_loss = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_val_loss.png'))
    im_ful_loss = Image.open(os.path.join(constants.RUNS, 'image_dumps', f'{filename}_ful_loss.png'))

    margin_x = 5
    margin_y = 5
    x = 320
    y = 240
    final = Image.new('RGB', ((x+margin_x)*4 + 60, (y+margin_y)*2), (255,255,255))
    final.paste(im_im, (margin_x//2, margin_y//2))
    final.paste(im_rnd_loss, (60+margin_x//2+x+margin_x, margin_y//2))
    final.paste(im_val_loss, (60+margin_x//2+(x+margin_x)*2, margin_y//2))
    final.paste(im_ful_loss, (60+margin_x//2+(x+margin_x)*3, margin_y//2))
    final.paste(im_tgt, (margin_x//2, y+margin_y+margin_y//2))
    final.paste(im_rnd, (60+margin_x//2+x+margin_x, y+margin_y+margin_y//2))
    final.paste(im_val, (60+margin_x//2+(x+margin_x)*2, y+margin_y+margin_y//2))
    final.paste(im_ful, (60+margin_x//2+(x+margin_x)*3, y+margin_y+margin_y//2))
    final.save(os.path.join(constants.RUNS, 'image_dumps', f'vis_{filename}.png'))

def visualize_image_target(image, target):
    from imageio import imwrite
    image_unnormalized = ((np.transpose(image, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
    target_mapped = (map_segmentation_to_colors(target.astype(np.uint8), 'matterport') * 255).astype(np.uint8)
    imwrite(os.path.join(constants.RUNS, 'image_dumps', f'{visualize_entropy.save_idx:04d}.png'), np.hstack((image_unnormalized, target_mapped)))
    visualize_entropy.save_idx += 1

def visualize_gt(dataset_name):
    from dataloader import dataset_base
    from dataloader.indoor_scenes import IndoorScenes
    from torch.utils.data import DataLoader
    import random 
    lmdb_handle = dataset_base.LMDBHandle(os.path.join(constants.HDD_DATASET_ROOT, dataset_name, "dataset.lmdb"), False)
    train_set = IndoorScenes(dataset_name, lmdb_handle, (240, 320), 'train')    
    ctr = 0 
    list_of_indices = list(range(len(train_set)))
    random.shuffle(list_of_indices)
    for i in list_of_indices:
        sample = train_set[i]
        visualize_image_target(sample['image'].numpy(), sample['label'].numpy())
        ctr+=1
        if ctr==1000:
            break
