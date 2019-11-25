import os
import torch
import constants
from utils.misc import get_learning_rate
from utils.summary import TensorboardSummary
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weights_labels
from torch.utils.data import DataLoader
import numpy as np
from utils.metrics import Evaluator
from tqdm import tqdm
import random


class Trainer:

    def __init__(self, args, model, train_set, val_set, test_set, class_weights, saver):
        self.args = args
        self.saver = saver
        self.saver.save_experiment_config()
        self.train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        self.val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        self.test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        self.train_summary = TensorboardSummary(os.path.join(self.saver.experiment_dir, "train"))
        self.train_writer = self.train_summary.create_summary()
        self.val_summary = TensorboardSummary(os.path.join(self.saver.experiment_dir, "validation"))
        self.val_writer = self.val_summary.create_summary()
        self.model = model
        self.dataset_size = {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)}

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        if args.use_balanced_weights:
            weight = torch.from_numpy(class_weights.astype(np.float32))
        else:
            weight = None

        if args.optimizer == 'SGD':
            print('Using SGD')
            self.optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            print('Using Adam')
            self.optimizer = torch.optim.Adam(train_params, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

        self.lr_scheduler = None
        if args.use_lr_scheduler:
            if args.lr_scheduler == 'step':
                print('Using step lr scheduler')                
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(x) for x in args.step_size.split(",")], gamma=0.1)

        self.criterion = SegmentationLosses(weight=weight, ignore_index=255, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.evaluator = Evaluator(train_set.num_classes)
        self.best_pred = 0.0

    def training(self, epoch):

        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_dataloader)
        tbar = tqdm(self.train_dataloader, desc='\r')

        visualization_index = int(random.random() * len(self.train_dataloader))
        vis_img, vis_tgt, vis_out = None, None, None

        self.train_writer.add_scalar('learning_rate', get_learning_rate(self.optimizer), epoch)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.train_writer.add_scalar('total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if i == visualization_index:
                vis_img, vis_tgt, vis_out = image, target, output

        self.train_writer.add_scalar('total_loss_epoch', train_loss / self.dataset_size['train'], epoch)
        if constants.VISUALIZATION:
            self.train_summary.visualize_state(self.train_writer, self.args.dataset, vis_img, vis_tgt, vis_out, epoch)

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('BestPred: %.3f' % self.best_pred)

    def validation(self, epoch, test=False):
        self.model.eval()
        self.evaluator.reset()
        
        ret_list = []
        if test:
            tbar = tqdm(self.test_dataloader, desc='\r')
        else:
            tbar = tqdm(self.val_dataloader, desc='\r')
        test_loss = 0.0

        visualization_index = int(random.random() * len(self.val_dataloader))
        vis_img, vis_tgt, vis_out = None, None, None

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            if i == visualization_index:
                vis_img, vis_tgt, vis_out = image, target, output

            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = torch.argmax(output, dim=1).data.cpu().numpy()
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)
            
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        mIoU_20 = self.evaluator.Mean_Intersection_over_Union_20()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        if not test:
            self.val_writer.add_scalar('total_loss_epoch', test_loss / self.dataset_size['val'], epoch)
            self.val_writer.add_scalar('mIoU', mIoU, epoch)
            self.val_writer.add_scalar('mIoU_20', mIoU_20, epoch)
            self.val_writer.add_scalar('Acc', Acc, epoch)
            self.val_writer.add_scalar('Acc_class', Acc_class, epoch)
            self.val_writer.add_scalar('fwIoU', FWIoU, epoch)
            if constants.VISUALIZATION:
                self.val_summary.visualize_state(self.val_writer, self.args.dataset, vis_img, vis_tgt, vis_out, epoch)

        print("Test: " if test else "Validation:")
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, mIoU_20:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, mIoU_20, FWIoU))
        print('Loss: %.3f' % test_loss)

        if not test:
            new_pred = mIoU
            if new_pred > self.best_pred:
                self.best_pred = new_pred
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                })

        return test_loss, mIoU, mIoU_20, Acc, Acc_class, FWIoU#, ret_list

    def load_best_checkpoint(self):
        checkpoint = self.saver.load_checkpoint()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'=> loaded checkpoint - epoch {checkpoint["epoch"]})')
        return checkpoint["epoch"]
