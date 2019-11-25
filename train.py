import os
import torch
import argument_parser
import constants
from utils.saver import Saver
from utils.trainer import Trainer
from dataloader.indoor_scenes import IndoorScenes
from dataloader import dataset_base
from model.deeplab import DeepLab
from utils.summary import TensorboardSummary
from utils.calculate_weights import calculate_weights_labels

def main():
    
    # script for training a model using 100% train set

    args = argument_parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    lmdb_handle = dataset_base.LMDBHandle(os.path.join(constants.HDD_DATASET_ROOT, args.dataset, "dataset.lmdb"), args.memory_hog)
    train_set = IndoorScenes(args.dataset, lmdb_handle, args.base_size, 'train')
    val_set = IndoorScenes(args.dataset, lmdb_handle, args.base_size, 'val')
    test_set = IndoorScenes(args.dataset, lmdb_handle, args.base_size, 'test')
    train_set.make_dataset_multiple_of_batchsize(args.batch_size)

    model = DeepLab(num_classes=train_set.num_classes, backbone=args.backbone, output_stride=args.out_stride, sync_bn=args.sync_bn)
    model = model.cuda()

    class_weights = None
    if args.use_balanced_weights:
        class_weights = calculate_weights_labels(train_set)

    saver = Saver(args)
    trainer = Trainer(args, model, train_set, val_set, test_set, class_weights, Saver(args))
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    start_epoch = 0
    if args.resume:
        args.resume = os.path.join(constants.RUNS, args.dataset, args.resume, 'checkpoint.pth.tar')
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at {args.resume}")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        trainer.model.load_state_dict(checkpoint['state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.best_pred = checkpoint['best_pred']
        print(f'=> loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')

    lr_scheduler = trainer.lr_scheduler

    for epoch in range(start_epoch, args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
        if lr_scheduler:
            lr_scheduler.step()

    epoch = trainer.load_best_checkpoint()
    _, best_mIoU, best_mIoU_20, best_Acc, best_Acc_class, best_FWIoU = trainer.validation(epoch, test=True)

    writer.add_scalar('test/mIoU', best_mIoU, epoch)
    writer.add_scalar('test/mIoU_20', best_mIoU_20, epoch)
    writer.add_scalar('test/Acc', best_Acc, epoch)
    writer.add_scalar('test/Acc_class', best_Acc_class, epoch)
    writer.add_scalar('test/fwIoU', best_FWIoU, epoch)

    trainer.train_writer.close()
    trainer.val_writer.close()


if __name__ == '__main__':
    main()
