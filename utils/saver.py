import os
import shutil
import torch
from collections import OrderedDict
import glob
import constants
import json
import imageio
import numpy as np

class Saver:

    def __init__(self, args, suffix='', experiment_group=None, remove_existing=False):

        self.args = args

        if experiment_group == None:
            experiment_group = args.dataset

        self.experiment_dir = os.path.join(constants.RUNS, experiment_group, args.checkname, suffix)

        if remove_existing and os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)

        if not os.path.exists(self.experiment_dir):
            print(f'Creating dir {self.experiment_dir}')
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):

        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        return torch.load(filename)

    def save_experiment_config(self):

        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        arg_dictionary = vars(self.args)
        log_file.write(json.dumps(arg_dictionary, indent=4, sort_keys=True))
        log_file.close()

    def save_active_selections(self, paths, regional=False):
        if regional:
            Saver.save_masks(os.path.join(self.experiment_dir, "selections") , paths)
        else:
            filename = os.path.join(self.experiment_dir, 'selections.txt')
            with open(filename, 'w') as fptr:
                for p in paths:
                    fptr.write(p.decode('utf-8') + '\n')

    @staticmethod
    def save_masks(directory, paths):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for p in paths:
            imageio.imwrite(os.path.join(directory, p.decode('utf-8') +'.png'), (paths[p]*255).astype(np.uint8))
