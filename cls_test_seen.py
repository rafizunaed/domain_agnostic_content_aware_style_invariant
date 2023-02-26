"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch import nn
import time

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from cls_dataset import ThoraxDS_Test, get_valid_transforms
from cls_model import DenseNet121_IBN, DenseNet121_IBN_proposed
from cls_configs import all_configs, DEVICE, DATASET_ROOT_DIR
from cls_trainer_callbacks import set_random_state

def get_args():
    parser = ArgumentParser(description='test_unseen')
    parser.add_argument('--run_config', type=str, default='proposed_md_DA_chexpert_mimic')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=160)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--image_resize_dim', type=int, default=224)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--test_dict_path', type=str, default='./datasets/split_and_test_dicts/chexpert_mimic_split_info_dict.npy')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)
            
    # get configs
    run_config = args.run_config  
    configs = all_configs[run_config]
    weight_saving_path = configs['weight_saving_path']

    all_mean_auc = []
    all_clswise_auc = []
    
    for fold_number in range(args.n_folds):
        set_random_state(args.seed)
        print('Running fold-{} ....'.format(fold_number))
          
        # get dataloader
        test_dict = np.load(args.test_dict_path, allow_pickle=True).item()
        test_fpaths=test_dict[f'fold_{fold_number}_test_fpaths']
        test_mask_fpaths=test_dict[f'fold_{fold_number}_test_mask_fpaths']
        test_labels=test_dict[f'fold_{fold_number}_test_labels']
        test_dataset = ThoraxDS_Test(
                            datasets_root_dir=DATASET_ROOT_DIR,
                            fpaths=test_fpaths,
                            mask_fpaths=test_mask_fpaths,
                            labels=test_labels,
                            transform=get_valid_transforms(args.image_resize_dim),
                            apply_lung_mask_crop=configs['apply_lung_mask_crop'],
                            )              
        test_loader = DataLoader(
                            test_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.n_workers,
                            drop_last=False,
                            pin_memory=True,
                            )  
        
        all_targets = []
        all_probabilities = []
        
        if configs['method'] == 'baseline':
            print('using baseline method: DenseNet121-IBN ....')
            model = DenseNet121_IBN(args.num_classes)
        elif configs['method'] in ['proposed_md_DA',  'proposed_sd_DA']:
            print('using proposed method: DenseNet121-IBN with proposed framework ....')
            model = DenseNet121_IBN_proposed(args.num_classes, configs['apply_fl_srm'], configs['apply_l_ccr'], \
                                              configs['apply_l_scr'], configs['apply_gfe_loss'])

        checkpoint = torch.load(weight_saving_path+f'/fold{fold_number}/checkpoint_best_auc_fold{fold_number}.pth')
        print('fold {} loss score: {:.4f}'.format(fold_number, checkpoint['val_loss']))
        print('fold {} auc score: {:.4f}'.format(fold_number, checkpoint['val_auc']))
        model.load_state_dict(checkpoint['Model_state_dict'])
        model = model.to(DEVICE)
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
        model.eval()                  
        del checkpoint
        
        torch.set_grad_enabled(False)
        with torch.no_grad():
            for itera_no, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                images, targets = data
                images = images.to(DEVICE) 
                targets = targets.to(DEVICE)                   
                
                with torch.cuda.amp.autocast():
                    out = model(images)
                    
                all_targets.append(targets.cpu().data.numpy())              
                y_prob = out['logits'].cpu().detach().clone().float().sigmoid()
                all_probabilities.append(y_prob.numpy())
            
        all_targets = np.concatenate(all_targets)
        all_probabilities = np.concatenate(all_probabilities)
        
        auc = roc_auc_score(all_targets, all_probabilities)
        all_mean_auc.append(auc)
        print(f'fold{fold_number} auc score: {auc*100}')
        time.sleep(1)
        
        all_clswise_auc.append(roc_auc_score(all_targets, all_probabilities, average=None))

        
        
    all_mean_auc = np.array(all_mean_auc)
    all_clswise_auc = np.stack(all_clswise_auc)
    
    print('5-fold auc mean: {:.2f}, std: {:.2f}'.format(all_mean_auc.mean(0)*100, all_mean_auc.std(0)*100))
    
    all_clswise_auc_mean = np.array([float('{:.2f}'.format(x*100)) for x in all_clswise_auc.mean(0)])
    all_clswise_auc_std = np.array([float('{:.2f}'.format(x*100)) for x in all_clswise_auc.std(0)])
    print(all_clswise_auc_mean)
    print(all_clswise_auc_std)
    
if __name__ == '__main__':
    main()