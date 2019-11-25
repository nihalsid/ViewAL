import argparse
import torch
import random


def parse_args():
    parser = argparse.ArgumentParser(description="ViewAL")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: mobilenet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='scannet-sample',
                        choices=['scannet', 'scenenet-rgbd', 'matterport3d','scannet-sample'],
                        help='dataset name (default: scannet-sample)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=str, default="384,512",
                        help='base image size')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: True)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=['step'],
                        help='lr scheduler mode: (default: step)')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--step-size', type=str, default='35', help='step size for lr-step-scheduler')
    parser.add_argument('--use-lr-scheduler', default=False, help='use learning rate scheduler', action='store_true')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: -1)')
    # checking point
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval (default: 5) - record metrics every Nth iteration')
    parser.add_argument('--memory-hog', action='store_true', default=False, help='load the whole dataset in RAM if true')
    parser.add_argument('--no-overlap', action='store_true', default=False, help='No overlap check - should be False')
    parser.add_argument('--max-iterations', type=int, default=8, help='max active iterations')
    parser.add_argument('--active-selection-size', type=int, default=1750, help='active selection size')
    parser.add_argument('--region-size', type=int, default=65, help='window size for window region methods')
    parser.add_argument('--region-selection-mode', type=str, default='superpixel', help='use superpixels or windows as region selection mode')
    parser.add_argument('--view-entropy-mode', type=str, default='mc_dropout', choices=['soft', 'vote', 'mc_dropout'], help='probability estimate = softmax or vote or mcdropout')
    parser.add_argument('--active-selection-mode', type=str, default='random',
                        choices=['random', 'viewentropy_region', 'voteentropy_soft', 'voteentropy_region', 'softmax_entropy', 'softmax_confidence', 'softmax_margin', 'coreset', 'voteentropy_max_repr', 'viewmc_kldiv_region', 'ceal'])
    parser.add_argument('--view-prob-aggr', type=str, default='entropy')
    parser.add_argument('--superpixel-dir', type=str, default='superpixel', help='directory for supepixel maps inside the dataset raw root')
    parser.add_argument('--superpixel-coverage-dir', type=str, default='coverage_superpixel', help='directory for coverage maps inside the dataset raw root')
    parser.add_argument('--superpixel-overlap', type=float, default=0.25, help='superpixel overlap threshold')
    parser.add_argument('--start-entropy-threshold', type=float, default=0.005, help='ceal hyperparameter')
    parser.add_argument('--entropy-change-per-selection', type=float, default=0.00033, help='ceal hyperparameter')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.base_size = [int(s) for s in args.base_size.split(',')]
    # manual seeding
    if args.seed == -1:
        args.seed = int(random.random() * 2000)
    print('Using random seed = ', args.seed)
    print('ActiveSelector: ', args.active_selection_mode)
    return args
