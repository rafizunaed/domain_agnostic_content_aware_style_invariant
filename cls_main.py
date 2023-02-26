"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

from argparse import ArgumentParser

import warnings

import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from cls_trainer_baseline import ModelTrainerBaseline
from cls_trainer_proposed_md_sd_DA import ModelTrainerProposed_md_sd_DA

from cls_dataset import ThoraxDS, ThoraxDS_Test, get_train_transforms, get_valid_transforms
from cls_model import DenseNet121_IBN, DenseNet121_IBN_proposed
from cls_configs import all_configs, DATASET_ROOT_DIR, MINI_IMAGENET_ROOT_DIR
from cls_trainer_callbacks import set_random_state, AverageMeter, PrintMeter

warnings.filterwarnings('ignore')

def get_args():
    """
    get command line args
    """
    parser = ArgumentParser(description='train')
    parser.add_argument('--run_configs_list', type=str, nargs="*", default=['baseline_md_chexpert_mimic', 'baseline_md_chexpert_mimic_mask_crop', 'baseline_md_chexpert_mimic_mask_crop_il_srm', 'proposed_md_DA_chexpert_mimic'])
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=160)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--image_resize_dim', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    """
    main function
    """

    args = get_args()

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)

    # check if there are duplicate weight saving paths
    unique_paths = np.unique([ x[1]['weight_saving_path'] for x in all_configs.items() ])
    assert len(all_configs.keys()) == len(unique_paths)

    for config_name in args.run_configs_list:
        configs = all_configs[config_name]
        split_dict = np.load(configs['split_dict_path'], allow_pickle=True).item()

        for fold_number in range(args.n_folds):
            set_random_state(args.seed)
            # if fold_number != 4:
            #     continue
            print(f'Running fold-{fold_number} ....')

            train_fpaths = split_dict[f'fold_{fold_number}_train_fpaths']
            train_mask_fpaths = split_dict[f'fold_{fold_number}_train_mask_fpaths']
            train_labels = split_dict[f'fold_{fold_number}_train_labels']

            val_fpaths = split_dict[f'fold_{fold_number}_val_fpaths']
            val_mask_fpaths = split_dict[f'fold_{fold_number}_val_mask_fpaths']
            val_labels = split_dict[f'fold_{fold_number}_val_labels']

            if configs['method'] in ['baseline', 'proposed_md_DA',  'proposed_sd_DA']:
                train_dataset = ThoraxDS(
                                    datasets_root_dir=DATASET_ROOT_DIR,
                                    fpaths=train_fpaths,
                                    mask_fpaths=train_mask_fpaths,
                                    labels=train_labels,
                                    transform=get_train_transforms(args.image_resize_dim),
                                    apply_il_srm=configs['apply_il_srm'],
                                    apply_lung_mask_crop=configs['apply_lung_mask_crop'],
                                    mini_imagenet_root_dir=MINI_IMAGENET_ROOT_DIR,
                                    style_source=configs['style_source'],
                                    )
                
            val_dataset = ThoraxDS_Test(
                                datasets_root_dir=DATASET_ROOT_DIR,
                                fpaths=val_fpaths,
                                mask_fpaths=val_mask_fpaths,
                                labels=val_labels,
                                transform=get_valid_transforms(args.image_resize_dim),
                                apply_lung_mask_crop=configs['apply_lung_mask_crop'],
                                )

            train_loader = DataLoader(
                                train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.n_workers,
                                drop_last=True,
                                pin_memory=True,
                                )

            val_loader = DataLoader(
                                val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_workers,
                                drop_last=False,
                                pin_memory=True,
                                )

            if configs['method'] == 'baseline':
                print('using baseline method: DenseNet121-IBN ....')
                model = DenseNet121_IBN(args.num_classes)
            elif configs['method'] in ['proposed_md_DA',  'proposed_sd_DA']:
                print('using proposed method: DenseNet121-IBN with proposed framework ....')
                model = DenseNet121_IBN_proposed(args.num_classes, \
                                                  configs['apply_fl_srm'], \
                                                  configs['apply_l_ccr'], \
                                                  configs['apply_l_scr'], configs['apply_gfe_loss'], configs['fl_srm_level'])
            
            if configs['checkpoint_root_path'] is not None:
                print('loading checkpoint!')
                wpath = configs['checkpoint_root_path']
                checkpoint = torch.load(f'{wpath}/fold{fold_number}/checkpoint_best_auc_fold{fold_number}.pth')
                print('fold {} loss score: {:.4f}'.format(fold_number, checkpoint['val_loss']))
                print('fold {} auc score: {:.4f}'.format(fold_number, checkpoint['val_auc']))
                model.load_state_dict(checkpoint['Model_state_dict'])
                del checkpoint
            else:
                print('no checkpoint found!')

            if configs['method'] == 'baseline':
                trainer_args = {
                        'model': model,
                        'Loaders': [train_loader, val_loader],
                        'metrics': {
                            'loss': AverageMeter,
                            'auc': PrintMeter,
                            'cls_loss': AverageMeter,
                            },
                        'checkpoint_saving_path': configs['weight_saving_path'],
                        'lr': args.lr,
                        'fold': fold_number,
                        'epochsTorun': configs['epochs'],
                        'gpu_ids': args.gpu_ids,
                        }

                trainer = ModelTrainerBaseline(**trainer_args)
                trainer.fit()

            elif configs['method'] in ['proposed_md_DA',  'proposed_sd_DA']:
                trainer_args = {
                        'model': model,
                        'Loaders': [train_loader, val_loader],
                        'metrics': {
                            'loss': AverageMeter,
                            'auc': PrintMeter,
                            'cls_loss': AverageMeter,
                            'gfe_loss': AverageMeter,
                            'consistency_loss': AverageMeter,
                            },
                        'checkpoint_saving_path': configs['weight_saving_path'],
                        'lr': args.lr,
                        'fold': fold_number,
                        'epochsTorun': configs['epochs'],
                        'gpu_ids': args.gpu_ids,
                        'apply_l_ccr': configs['apply_l_ccr'],
                        'apply_l_scr': configs['apply_l_scr'],
                        'apply_gfe_loss': configs['apply_gfe_loss'],
                        }

                trainer = ModelTrainerProposed_md_sd_DA(**trainer_args)
                trainer.fit()

if __name__ == '__main__':
    main()
    