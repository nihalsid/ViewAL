# ViewAL: Active Learning with Viewpoint Entropy for Semantic Segmentation

This repository contains the implementation for the paper:

Yawar Siddiqui, Julien Valentin and Matthias Niessner, ["ViewAL: Active Learning with Viewpoint Entropy for Semantic Segmentation"](https://arxiv.org/abs/1911.11789) ([video](https://youtu.be/tAGdx2j-X_g))

![VisualizationGIF](https://user-images.githubusercontent.com/932110/69557468-f190ae00-0fa6-11ea-9321-309ba55da63d.gif)

## Running

#### Arguments

```
train_active.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                       [--out-stride OUT_STRIDE]
                       [--dataset {scannet,scenenet-rgbd,matterport3d,scannet-sample}]
                       [--workers N] [--base-size BASE_SIZE]
                       [--sync-bn SYNC_BN] [--loss-type {ce,focal}]
                       [--epochs N] [--batch-size N] [--use-balanced-weights]
                       [--lr LR] [--lr-scheduler {step}]
                       [--optimizer {SGD,Adam}] [--step-size STEP_SIZE]
                       [--use-lr-scheduler] [--momentum M] [--weight-decay M]
                       [--nesterov] [--gpu-ids GPU_IDS] [--seed S]
                       [--checkname CHECKNAME] [--eval-interval EVAL_INTERVAL]
                       [--memory-hog]
                       [--max-iterations MAX_ITERATIONS]
                       [--active-selection-size ACTIVE_SELECTION_SIZE]
                       [--region-size REGION_SIZE]
                       [--region-selection-mode REGION_SELECTION_MODE]
                       [--view-entropy-mode {soft,vote,mc_dropout}]
                       [--active-selection-mode {random,viewentropy_region,voteentropy_soft,voteentropy_region,softmax_entropy,softmax_confidence,softmax_margin,coreset,voteentropy_max_repr,viewmc_kldiv_region,ceal}]
                       [--superpixel-dir SUPERPIXEL_DIR]
                       [--superpixel-coverage-dir SUPERPIXEL_COVERAGE_DIR]
                       [--superpixel-overlap SUPERPIXEL_OVERLAP]
                       [--start-entropy-threshold START_ENTROPY_THRESHOLD]
                       [--entropy-change-per-selection ENTROPY_CHANGE_PER_SELECTION]
```

Run `--help` for more details.

#### Active Selection Modes

Apart from implementation of our method, we provide implementation of other popular active selection methods adapted for semantic segmentation.

| Option        | Method  |
| --------------- | ------------ |
|random| RAND selection |
|voteentropy_max_repr| MAXRPR selection|
|voteentropy_soft| MCDR selection|
|voteentropy_region| RMCDR selection|
|softmax_entropy| ENT selection|
|softmax_confidence| CONF selection |
|softmax_margin| MAR selection |
|coreset| CSET selection |
|ceal| CEAL selection |
|viewmc_kldiv_region| ViewAL selection|

For a description of the methods, check out appendix section of the paper.

#### Example commands


##### View AL
```sh
# sample dataset
python train_active.py --dataset scenenet-rgbd --workers 2 --epochs 50 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 35 --checkname regional_viewmckldiv_spx_1_7x2_lr-0.0004_bs-6_ep-60_wb-0_lrs-1_240x320 --base-size 240,320 --max-iterations 7 --active-selection-size 2 --active-selection-mode viewmc_kldiv_region --region-selection-mode superpixel

# scenenet-rgbd
python train_active.py --dataset scenenet-rgbd --workers 2 --epochs 50 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 35 --checkname regional_viewmckldiv_spx_1_7x1500_lr-0.0004_bs-6_ep-60_wb-0_lrs-1_240x320 --base-size 240,320 --max-iterations 7 --active-selection-size 1500 --active-selection-mode viewmc_kldiv_region --region-selection-mode superpixel
```

##### Random
```sh
python train_active.py --dataset scenenet-rgbd --workers 2 --epochs 50 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 35 --checkname random_0_7x1500_lr-0.0004_bs-6_ep-60_wb-0_lrs-0_240x320 --base-size 240,320 --max-iterations 7 --active-selection-size 1500 --active-selection-mode random
```

##### Softmax Entropy
```sh
python train_active.py --dataset scenenet-rgbd --workers 2 --epochs 50 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 35 --checkname softmax_entropy_0_7x1500_lr-0.0004_bs-6_ep-50_wb-0_lrs-1_240x320 --base-size 240,320 --max-iterations 7 --active-selection-size 1500 --active-selection-mode softmax_entropy
```

##### Regional MCDR
```sh
python train_active.py --dataset scenenet-rgbd --workers 2 --epochs 50 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 35 --checkname regional_voteentropy_window_0_7x1500_lr-0.0004_bs-6_ep-60_wb-0_lrs-1_240x320 --base-size 240,320 --max-iterations 7 --active-selection-size 1500 --active-selection-mode voteentropy_region --region-selection-mode window
```

##### CEAL 
```sh
python train_active.py --dataset scenenet-rgbd --workers 2 --epochs 50 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 35 --checkname ceal-0.00275_7x1500_lr-0.0005_bs-6_ep-50_wb-0_lrs-1_240x320 --max-iterations 7 --active-selection-size 1500 --base-size 240,320 --active-selection-mode ceal --start-entropy-threshold 0.0275 --entropy-change-per-selection 0.001815
```

##### MCDR
```sh
python train_active.py --dataset scenenet-rgbd --workers 2 --epochs 50 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 35 --checkname mcdropoutentropy_0_7x1500_lr-0.0004_bs-6_ep-50_wb-0_lrs-1_240x320 --base-size 240,320 --max-iterations 7 --active-selection-size 1500 --active-selection-mode voteentropy_soft
```

##### Full training 
```sh
python train.py --dataset scenenet-rgbd --workers 2 --epochs 70 --eval-interval 5 --batch-size=6 --lr 0.0004 --use-lr-scheduler --lr-scheduler step --step-size 40 --checkname full-run_0_lr-0.0004_bs-6_ep-60_wb-0_lrs-0_240x320 --base-size 240,320
```

## Files

Overall code structure is as follows: 

| File / Folder | Description |
| ------------- |-------------| 
| train_active.py | Training script for active learning methods | 
| train.py | Training script for full dataset training | 
| constants.py | Constants used across the code |
| argument_parser.py | Arguments parsing code |
| active_selection | Implementation of our method and other active learning methods for semantic segmentation |
| dataloader | Dataset classes |
| dataset | Train/test splits and raw files of datasets |
| model | DeeplabV3+ implementation (inspired from [here](https://github.com/jfzhang95/pytorch-deeplab-xception))|
| utils| Misc utils |

The datasets must follow the following structure

```
dataset # root dataset directory
├── dataset-name
    ├── raw
        ├── selections
            ├── color # rgb frames
            ├── label # ground truth maps
            ├── depth # depth maps
            ├── pose # camera extrinsics for each frame
            ├── info # camera intrinsics
            ├── superpixel # superpixel maps
            ├── coverage_superpixel # coverage maps
    ├── selections
        ├── seedset_0_frames.txt # seed set
        ├── train_frames.txt 
        ├── val_frames.txt
        ├── test_frames.txt
    ├── dataset.lmdb # rgb frames + labels in lmdb format
```

A small example dataset is provided with this repository in [`dataset/scannet-sample`](https://github.com/nihalsid/ViewAL/tree/master/dataset/scannet-sample).

## Data Generation

To use this repository datasets must be in the structure described in last section. For creating the lmdb database, seed set, train / test splits and superpixel maps check helper scripts in [`dataset/preprocessing-scripts`](https://github.com/nihalsid/ViewAL/tree/master/dataset/preprocessing-scripts). We use [this SEEDS implementation](https://github.com/davidstutz/seeds-revised) for generating superpixels. Further, to generate superpixel coverage maps (`coverage_superpixel`) check [`utils/superpixel_projections.py`](https://github.com/nihalsid/ViewAL/blob/master/utils/superpixel_projections.py). 

## Citation

If you use this code, please cite the paper:

```
@misc{siddiqui2019viewal,
    title={ViewAL: Active Learning with Viewpoint Entropy for Semantic Segmentation},
    author={Yawar Siddiqui and Julien Valentin and Matthias Nießner},
    year={2019},
    eprint={1911.11789},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
