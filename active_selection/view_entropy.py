import numpy as np
from numpy.linalg import inv
from utils.misc import visualize_point_cloud, visualize_entropy, turn_on_dropout
import constants
from dataloader import indoor_scenes
from collections import OrderedDict, defaultdict
from model.deeplab import DeepLab
import torch
from PIL import Image
from tqdm import tqdm


class ViewEntropySelector:

    def __init__(self, dataset, lmdb_handle, superpixel_dir, base_size, num_classes, scoring_func, mc_dropout, superpixel_averaged_maxed, return_non_reduced_maps=False):
        self.lmdb_handle = lmdb_handle
        self.base_size = base_size
        self.superpixel_dir = superpixel_dir
        self.dataset = dataset
        self.num_classes = num_classes
        self.superpixel_averaged_maxed = superpixel_averaged_maxed
        self.return_non_reduced_maps = return_non_reduced_maps
        self.class_weights = None
        self.softmax = torch.nn.Softmax2d()
        self.mc_dropout = mc_dropout
        self.scoring_func_name = scoring_func
        if scoring_func == 'entropy':
            self.scoring_function = self.entropy_function
        elif scoring_func == 'kldiv':
            self.scoring_function = self.kldiv_function
        else:
            raise NotImplementedError

    def calculate_scores(self, model, images, precomputed_probabilities=None, save_probabilites=False):
        dataset = indoor_scenes.IndoorScenesWithAllInfo(self.dataset, self.lmdb_handle, self.superpixel_dir, self.base_size, images)
        scene_id_to_index = dataset.scene_id_to_index
        scores = []
        image_paths = []
        superpixel_masks = []
        all_frame_coverages = OrderedDict()
        saved_probabilities = []

        # turn on dropout if mc dropout probabilities used
        if self.mc_dropout:
            model.eval()
            model.apply(turn_on_dropout)

        for scene_ctr, scene_id in enumerate(tqdm(scene_id_to_index, desc='ViewEntropy[Scene]')):
            depths = []
            poses = []
            num_frames_scenes = len(scene_id_to_index[scene_id])
            probabilities_on_gpu = False if num_frames_scenes > 450 else True
            probabilities_type = torch.cuda if probabilities_on_gpu else torch
            if not precomputed_probabilities:
                probabilities = torch.zeros(len(scene_id_to_index[scene_id]) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT, self.num_classes).type(probabilities_type.FloatTensor)
            else:
                probabilities = precomputed_probabilities[scene_ctr].type(probabilities_type.FloatTensor)
            intrinsic = None

            # get predictions for all candidates
            for im_idx, frame_id in enumerate(tqdm(scene_id_to_index[scene_id], desc='ViewEntropy[Pred]')):
                sample = dataset[frame_id]
                depths.append(sample['depth'])
                poses.append(sample['pose'])
                intrinsic = sample['intrinsic']

                image = sample['image'].unsqueeze(0).cuda()

                if not precomputed_probabilities:
                    output = torch.zeros(self.num_classes, constants.DEPTH_HEIGHT, constants.DEPTH_WIDTH).type(probabilities_type.FloatTensor)
                    if self.mc_dropout:
                        with torch.no_grad():
                            for step in range(constants.MC_STEPS):
                                output += self.softmax(model(image))[0].type(probabilities_type.FloatTensor)
                        output /= constants.MC_STEPS
                    else:
                        with torch.no_grad():
                            output = self.softmax(model(image))[0].type(probabilities_type.FloatTensor)

                    probabilities[im_idx * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT: (im_idx + 1) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT, :] = output.permute(1, 2, 0).view(-1, self.num_classes)
   
            # project all prediction probabilities to world
            world_coordinates, frame_origins = self.project_images_to_world(depths, poses, intrinsic, scene_id_to_index[scene_id])

            # project prediction probabilities to each frame
            for ctr, frame_id in enumerate(tqdm(scene_id_to_index[scene_id], desc='ViewEntropy[Entropy]')):
                sample = dataset[frame_id]
                score, frame_coverages, _ = self.get_projected_prediction_entropy(frame_id, sample['depth'], sample['pose'], sample['intrinsic'], world_coordinates, probabilities, frame_origins, dataset.num_classes, probabilities_type)
                if not score is None:
                    score[sample['label'].numpy() == 255] = 0
                    all_frame_coverages[frame_id] = frame_coverages
                    if self.superpixel_averaged_maxed:
                        superpixels = sample['superpixel']
                        superpixels = np.asarray(Image.fromarray(superpixels).resize((constants.DEPTH_WIDTH, constants.DEPTH_HEIGHT), Image.NEAREST))
                        unique_superpixels_as_list = np.unique(superpixels).tolist()
                        score_per_superpixel = defaultdict(int)
                        for spx_id in unique_superpixels_as_list:
                            spx_mean = score[superpixels == spx_id].mean()
                            score_per_superpixel[spx_id] = spx_mean
                            score[superpixels == spx_id] = spx_mean
                        if self.return_non_reduced_maps:
                            scores.append(score_per_superpixel)
                        else:
                            scores.append(score.max())
                    else:
                        if self.return_non_reduced_maps:
                            scores.append(score)
                        else:
                            scores.append(score.sum())

                    image_paths.append(images[frame_id])
                    if self.return_non_reduced_maps:
                        superpixel_masks.append(sample['superpixel'])

            if save_probabilites:
                saved_probabilities.append(probabilities.cpu())

            del world_coordinates, probabilities, frame_origins
            del depths, poses
            torch.cuda.empty_cache()

        return scores, image_paths, all_frame_coverages, superpixel_masks, saved_probabilities

    def project_image_to_world(self, x, y, depth, cam2world, depth_intrinsic):
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

    def project_images_to_world(self, depths, cam2worlds, depth_intrinsic, frames):

        x = np.linspace(0, constants.DEPTH_WIDTH - 1, constants.DEPTH_WIDTH)
        y = np.linspace(0, constants.DEPTH_HEIGHT - 1, constants.DEPTH_HEIGHT)
        x_mesh, y_mesh = np.meshgrid(x, y)

        world_coordinates = torch.zeros(4, len(depths) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT).type(torch.cuda.FloatTensor)
        
        frame_origins = torch.zeros(len(depths) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT).type(torch.cuda.IntTensor)

        for im_idx in tqdm(range(len(depths)), desc='ViewEntropy[World]'):

            world_coordinates[:, im_idx * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT: (im_idx + 1) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT] = self.project_image_to_world(torch.from_numpy(x_mesh).type(torch.cuda.FloatTensor).flatten(), torch.from_numpy(y_mesh).type(
                torch.cuda.FloatTensor).flatten(), torch.from_numpy(depths[im_idx][:]).type(torch.cuda.FloatTensor).flatten(), cam2worlds[im_idx], depth_intrinsic)
            frame_origins[im_idx * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT: (im_idx + 1) * constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT] = torch.ones(constants.DEPTH_WIDTH * constants.DEPTH_HEIGHT).type(torch.cuda.IntTensor) * frames[im_idx]

        return world_coordinates, frame_origins

    def get_projected_prediction_entropy(self, destination_frame_index, depth, cam2world, depth_intrinsic, world_coordinates, probabilities, frame_origins, num_classes, probabilities_type):
        world_coordinates_copy = world_coordinates.transpose(0, 1)[:, :3]
        frame_origins_copy = frame_origins.clone()
        projected_points = torch.mm(torch.mm(torch.from_numpy(depth_intrinsic).type(torch.cuda.FloatTensor),
                                             torch.from_numpy(inv(cam2world)).type(torch.cuda.FloatTensor)), world_coordinates)
        projected_points = projected_points.transpose(0, 1)[:, :3]
        projected_points[:, 0] /= projected_points[:, 2]
        projected_points[:, 1] /= projected_points[:, 2]
        projected_points[:, 2] /= projected_points[:, 2]
        projected_points = torch.round(projected_points)
        selection_mask_0 = ~torch.isnan(projected_points[:, 2]) & (projected_points[:, 0] >= 0) & (projected_points[:, 0] < constants.DEPTH_WIDTH) & (
            projected_points[:, 1] >= 0) & (projected_points[:, 1] < constants.DEPTH_HEIGHT)

        projected_points = projected_points[selection_mask_0, :2]
        world_coordinates_copy = world_coordinates_copy[selection_mask_0]
        frame_origins_selected = frame_origins_copy[selection_mask_0]

        depth = torch.from_numpy(depth).type(torch.cuda.FloatTensor)
        depth = depth[projected_points[:, 1].type(torch.cuda.LongTensor), projected_points[:, 0].type(torch.cuda.LongTensor)].flatten()
        
        backprojected_points = self.project_image_to_world(projected_points[:, 0], projected_points[
            :, 1], depth, cam2world, depth_intrinsic).transpose(0, 1)[:, :3]

        selection_mask_1 = (torch.norm(world_coordinates_copy - backprojected_points, dim=1) < constants.WORLD_DISTANCE_THRESHOLD)

        selection_mask_0[selection_mask_0] = selection_mask_0[selection_mask_0] & selection_mask_1 

        projected_points = projected_points[selection_mask_1]
        projected_points_flat = (projected_points[:, 1] * constants.DEPTH_WIDTH + projected_points[:, 0]).type(torch.cuda.LongTensor)
        frame_origins_selected = frame_origins_selected[selection_mask_1]

        unique_frame_indices, frame_index_counts = torch.unique(frame_origins_selected, return_counts=True)
        frame_index_counts = frame_index_counts.cpu().numpy().tolist()
        unique_frame_indices = unique_frame_indices.cpu().numpy().tolist()

        coverage_dict = {}

        for i in range(len(unique_frame_indices)):
            coverage_dict[unique_frame_indices[i]] = frame_index_counts[i] / (constants.DEPTH_HEIGHT * constants.DEPTH_WIDTH)

        if not projected_points.shape[0] == 0:
            return_map, return_mask = self.scoring_function(destination_frame_index, selection_mask_0, probabilities, probabilities_type, projected_points_flat, frame_origins_copy)
            return return_map, coverage_dict, return_mask
        return None, None, None

    def entropy_function(self, destination_frame_index, selection_mask_0, probabilities, probabilities_type, projected_points_flat, frame_origins):
        # view entropy score
        probability_matrix = torch.zeros((constants.DEPTH_HEIGHT * constants.DEPTH_WIDTH, self.num_classes)).type(torch.cuda.FloatTensor)
        selection_mask_0 = selection_mask_0.type(probabilities_type.BoolTensor)
        probability_matrix.index_add_(0, projected_points_flat, probabilities[selection_mask_0].cuda())
        probability_matrix = probability_matrix / probability_matrix.sum(dim=1).view(-1, 1)
        return_mask = (probability_matrix != probability_matrix).cpu().numpy()
        probability_matrix[ probability_matrix != probability_matrix ] = 0
        entropy_map = torch.zeros((constants.DEPTH_HEIGHT * constants.DEPTH_WIDTH)).type(torch.cuda.FloatTensor)
        for c in range(self.num_classes):
            entropy_map = entropy_map - (probability_matrix[:, c] * torch.log2(probability_matrix[:, c] + 1e-12))
        entropy_map = entropy_map.view(constants.DEPTH_HEIGHT, constants.DEPTH_WIDTH)
        return_map = entropy_map.cpu().numpy()
        del probability_matrix
        torch.cuda.empty_cache()
        return return_map, return_mask

    def kldiv_function(self, destination_frame_index, selection_mask_0, probabilities, probabilities_type, projected_points_flat, frame_origins):
        # view divergence score
        sum_log_Q_P = torch.zeros((constants.DEPTH_HEIGHT * constants.DEPTH_WIDTH, self.num_classes)).type(torch.cuda.FloatTensor)
        divisor = torch.zeros(constants.DEPTH_HEIGHT * constants.DEPTH_WIDTH).type(torch.cuda.FloatTensor)
        selection_mask_0 = selection_mask_0.type(probabilities_type.BoolTensor)
        P_sub = probabilities.cuda()[frame_origins == destination_frame_index]
        assert(P_sub.shape[0] == constants.DEPTH_HEIGHT * constants.DEPTH_WIDTH and P_sub.shape[1] == self.num_classes)
        log_Q_P = torch.log2(torch.div(probabilities[selection_mask_0].cuda() + 1e-8, P_sub[projected_points_flat] + 1e-8))
        sum_log_Q_P.index_add_(0, projected_points_flat, log_Q_P)
        divisor.index_add_(0, projected_points_flat, torch.ones_like(projected_points_flat).type(torch.cuda.FloatTensor))
        sum_log_Q_P = sum_log_Q_P / divisor.view(-1, 1)
        return_mask = (sum_log_Q_P != sum_log_Q_P).cpu().numpy()
        sum_log_Q_P[ sum_log_Q_P != sum_log_Q_P ] = 0
        return_map = (-1 * torch.mul(P_sub, sum_log_Q_P).sum(dim=1)).view(constants.DEPTH_HEIGHT, constants.DEPTH_WIDTH).cpu().numpy()
        del sum_log_Q_P, divisor
        torch.cuda.empty_cache()
        return return_map, return_mask
