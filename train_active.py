import os
import torch
import argument_parser
import constants
from utils.saver import Saver
from utils.trainer import Trainer
from dataloader.indoor_scenes import IndoorScenes
from dataloader.indoor_scenes import get_active_dataset
from dataloader import dataset_base
from active_selection import get_active_selector
from model.deeplab import DeepLab
from utils.summary import TensorboardSummary
from utils.calculate_weights import calculate_weights_labels


def main():

    args = argument_parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # hardcoding scannet

    # get handle to lmdb dataset
    lmdb_handle = dataset_base.LMDBHandle(os.path.join(constants.HDD_DATASET_ROOT, args.dataset, "dataset.lmdb"), args.memory_hog)
    
    # create train val and test sets
    train_set = get_active_dataset(args.active_selection_mode)(args.dataset, lmdb_handle, args.superpixel_dir, args.base_size, 'seedset_0')
    val_set = IndoorScenes(args.dataset, lmdb_handle, args.base_size, 'val')
    test_set = IndoorScenes(args.dataset, lmdb_handle, args.base_size, 'test')

    class_weights = None
    if args.use_balanced_weights:
        class_weights = calculate_weights_labels(get_active_dataset(args.active_selection_mode)(args.dataset, lmdb_handle, args.superpixel_dir, args.base_size, 'train'))

    saver = Saver(args)
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    # get active selection method
    active_selector = get_active_selector(args, lmdb_handle, train_set)


    # for each active selection iteration
    for selection_iter in range(args.max_iterations):

        fraction_of_data_labeled = int(round(train_set.get_fraction_of_labeled_data() * 100))
        
        if os.path.exists(os.path.join(constants.RUNS, args.dataset, args.checkname, f'runs_{fraction_of_data_labeled:03d}', "selections")):
            # resume: load selections if this is a rerun, and selections are available from a previous run
            train_set.load_selections(os.path.join(constants.RUNS, args.dataset, args.checkname, f'runs_{fraction_of_data_labeled:03d}', "selections"))
        elif os.path.exists(os.path.join(constants.RUNS, args.dataset, args.checkname, f'runs_{fraction_of_data_labeled:03d}', "selections.txt")):
            # resume: load selections if this is a rerun, and selections are available from a previous run
            train_set.load_selections(os.path.join(constants.RUNS, args.dataset, args.checkname, f'runs_{fraction_of_data_labeled:03d}', "selections.txt"))
        else:
            # active selection iteration

            train_set.make_dataset_multiple_of_batchsize(args.batch_size)
            # create model from scratch
            model = DeepLab(num_classes=train_set.num_classes, backbone=args.backbone, output_stride=args.out_stride, sync_bn=args.sync_bn,
                            mc_dropout=((args.active_selection_mode.startswith('viewmc')) or(args.active_selection_mode.startswith('vote')) or args.view_entropy_mode == 'mc_dropout'))
            model = model.cuda()
            # create trainer
            trainer = Trainer(args, model, train_set, val_set, test_set, class_weights, Saver(args, suffix=f'runs_{fraction_of_data_labeled:03d}'))
            
            # train for args.epochs epochs
            lr_scheduler = trainer.lr_scheduler
            for epoch in range(args.epochs):
                trainer.training(epoch)
                if epoch % args.eval_interval == (args.eval_interval - 1):
                    trainer.validation(epoch)
                if lr_scheduler:
                    lr_scheduler.step()

            train_set.reset_dataset()
            epoch = trainer.load_best_checkpoint()

            # get best val miou / metrics
            _, best_mIoU, best_mIoU_20, best_Acc, best_Acc_class, best_FWIoU = trainer.validation(epoch, test=True)

            trainer.evaluator.dump_matrix(os.path.join(trainer.saver.experiment_dir, "confusion_matrix.npy"))

            writer.add_scalar('active_loop/mIoU', best_mIoU, train_set.get_fraction_of_labeled_data() * 100)
            writer.add_scalar('active_loop/mIoU_20', best_mIoU_20, train_set.get_fraction_of_labeled_data() * 100)
            writer.add_scalar('active_loop/Acc', best_Acc, train_set.get_fraction_of_labeled_data() * 100)
            writer.add_scalar('active_loop/Acc_class', best_Acc_class, train_set.get_fraction_of_labeled_data() * 100)
            writer.add_scalar('active_loop/fwIoU', best_FWIoU, train_set.get_fraction_of_labeled_data() * 100)

            # make active selection
            active_selector.select_next_batch(model, train_set, args.active_selection_size)
            # save selections
            trainer.saver.save_active_selections(train_set.get_selections(), args.active_selection_mode.endswith("_region"))
            trainer.train_writer.close()
            trainer.val_writer.close()

        print(selection_iter, " / Train-set length: ", len(train_set))
        
    writer.close()

if __name__ == '__main__':
    main()
