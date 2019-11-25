from tqdm import tqdm
import os
from pathlib import Path
import random
from multiprocessing import Pool
from skimage.io import imread
    
raw_files_location = "ViewAL/dataset/scannet-sample/raw/selections"
seeds_segmentor_path = "seeds-revised/bin/Release/reseeds_cli.exe"
splits_path = Path('ViewAL/dataset/scannet-sample/selections')

def read_scene_list(path):
    with open(path, "r") as fptr:
        return [x.strip() for x in fptr.readlines() if x.strip() != ""]

def write_frames_list(name, paths):
    with open((splits_path / f"{name}_frames.txt"), "w") as fptr:
        for x in paths:
            fptr.write(f"{x}\n")

def create_splits():

    train_scenes = read_scene_list(str(splits_path / "scenes_train.txt"))
    val_scenes = read_scene_list(str(splits_path / "scenes_val.txt"))
    test_scenes = read_scene_list(str(splits_path / "scenes_test.txt"))

    train_frames = []
    val_frames = []
    test_frames = []

    for x in (raw_files_location / "color").iterdir():
        # For scannet names - format = {sceneid_rescanid}_frameid
        if "_".join(x.name.split("_")[0:2]) in train_scenes:
            train_frames.append(x.name.split(".")[0])
        elif "_".join(x.name.split("_")[0:2]) in val_scenes:
            val_frames.append(x.name.split(".")[0])
        elif "_".join(x.name.split("_")[0:2]) in test_scenes:
            test_frames.append(x.name.split(".")[0])

    print(len(train_frames), len(val_frames), len(test_frames))

    write_frames_list("train", train_frames)
    write_frames_list("val", val_frames)
    write_frames_list("test", test_frames)

def call(*popenargs, **kwargs):
    from subprocess import Popen, PIPE
    kwargs['stdout'] = PIPE
    kwargs['stderr'] = PIPE
    p = Popen(popenargs, **kwargs)
    stdout, stderr = p.communicate()
    if stdout:
        for line in stdout.decode("utf-8").strip().split("\n"):
            print(line)
    if stderr:
        for line in stderr.decode("utf-8").strip().split("\n"):
            print(line)
    return p.returncode

def create_superpixel_segmentations():
    all_args = "--spatial-weight 0.2 --superpixels 40 --iterations 10 --confidence 0.001"
    colordir_src = os.path.join(raw_files_location, "color")
    args = all_args.split(" ")
    with Pool(processes=8) as pool:
        arg_list = []
        for c in tqdm(os.listdir(colordir_src)):
            color_base_name = c.split(".")[0]
            spx_target = os.path.join(raw_files_location, "superpixel", f"{color_base_name}.png")
            arg_list.append((seeds_segmentor_path, "--input", os.path.join(colordir_src, c), "--output", spx_target,
                             args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], "--index"))
        pool.starmap(call, tqdm(arg_list))


def create_split_mixed():
    list_of_frames = [x.split(".")[0] for x in os.listdir(os.path.join(raw_files_location, "color"))]
    train_frames = [list_of_frames[i] for i in random.sample(range(len(list_of_frames)), int(0.60 * len(list_of_frames)))]
    remaining_frames = [x for x in list_of_frames if x not in train_frames]
    val_frames = [remaining_frames[i] for i in random.sample(range(len(remaining_frames)), int(0.15 * len(list_of_frames)))]
    test_frames = [x for x in remaining_frames if x not in val_frames]
    print(len(train_frames)/len(list_of_frames), len(val_frames)/len(list_of_frames), len(test_frames)/len(list_of_frames))
    with open(splits_path / "train_frames.txt", "w") as fptr:
        for x in train_frames:
            fptr.write(x+"\n")
    with open(splits_path / "val_frames.txt", "w") as fptr:
        for x in val_frames:
            fptr.write(x+"\n")
    with open(splits_path / "test_frames.txt", "w") as fptr:
        for x in test_frames:
            fptr.write(x+"\n")

def create_seed_set():
    train_frames = (splits_path / "train_frames.txt").read_text().split()
    seed_frames = [train_frames[i] for i in random.sample(range(len(train_frames)), int(0.05 * len(train_frames)))]
    print(len(seed_frames))
    with open(splits_path / "seedset_0_frames.txt", "w") as fptr:
        for x in seed_frames:
            fptr.write(x.split(".")[0]+"\n")

if __name__=='__main__':
    create_superpixel_segmentations()
    create_split_mixed() # or create_split()
    create_seed_set()
    