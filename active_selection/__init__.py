from active_selection import random_selection, view_entropy, vote_entropy,  softmax_entropy, regional_vote_entropy, softmax_confidence, softmax_margin, core_set, max_repr, regional_view_entropy_kl, ceal
from dataloader.dataset_base import OverlapHandler
import constants
import os

def get_active_selector(args, lmdb_handle, train_set):
    if args.active_selection_mode == 'random':
        return random_selection.RandomSelector()
    elif args.active_selection_mode == 'ceal':
        return ceal.CEALSelector(args.dataset, lmdb_handle, args.base_size, args.batch_size, train_set.num_classes, args.start_entropy_threshold, args.entropy_change_per_selection)
    elif args.active_selection_mode == 'voteentropy_soft':
        return vote_entropy.VoteEntropySelector(args.dataset, lmdb_handle, args.base_size, args.batch_size, train_set.num_classes, True)
    elif args.active_selection_mode == 'softmax_entropy':
        return softmax_entropy.SoftmaxEntropySelector(args.dataset, lmdb_handle, args.base_size, args.batch_size, train_set.num_classes)
    elif args.active_selection_mode == 'softmax_margin':
        return softmax_margin.SoftmaxMarginSelector(args.dataset, lmdb_handle, args.base_size, args.batch_size)
    elif args.active_selection_mode == 'softmax_confidence':
        return softmax_confidence.SoftmaxConfidenceSelector(args.dataset, lmdb_handle, args.base_size, args.batch_size)
    elif args.active_selection_mode == 'voteentropy_region':
        overlap_handler = None
        if not args.no_overlap:
            overlap_handler = OverlapHandler(os.path.join(constants.SSD_DATASET_ROOT, args.dataset, "raw", "selections", args.superpixel_coverage_dir), args.superpixel_overlap, memory_hog_mode=True)
        return regional_vote_entropy.RegionalVoteEntropySelector(args.dataset, lmdb_handle, args.superpixel_dir, args.base_size, args.batch_size, train_set.num_classes, args.region_size, overlap_handler, mode=args.region_selection_mode)
    elif args.active_selection_mode == 'coreset':
        return core_set.CoreSetSelector(args.dataset, lmdb_handle, args.base_size, args.batch_size)
    elif args.active_selection_mode == 'voteentropy_max_repr':
        return max_repr.MaxRepresentativeSelector(args.dataset, lmdb_handle, args.base_size, args.batch_size, train_set.num_classes)
    elif args.active_selection_mode == 'viewmc_kldiv_region':
        overlap_handler = None
        if not args.no_overlap:
            overlap_handler = OverlapHandler(os.path.join(constants.SSD_DATASET_ROOT, args.dataset, "raw", "selections", args.superpixel_coverage_dir), args.superpixel_overlap, memory_hog_mode=True)
        return regional_view_entropy_kl.RegionalViewEntropyWithKldivSelector(args.dataset, lmdb_handle, args.superpixel_dir, args.base_size, train_set.num_classes, args.region_size, overlap_handler, mode=args.region_selection_mode)
    raise NotImplementedError
