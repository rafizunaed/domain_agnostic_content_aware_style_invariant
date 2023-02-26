"""
version: 19 February, 2023
Programmed by Mohammad Zunaed
"""

import os
import numpy as np
from argparse import ArgumentParser
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def get_args():
    parser = ArgumentParser(description='create cv splits')
    parser.add_argument('--seed', type=int, default=4690)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--mimic_label_info_path', type=str, default='./mimic/mimic_label_info.npy')
    parser.add_argument('--chexpert_label_info_path', type=str, default='./chexpert/chexpert_label_info.npy')
    parser.add_argument('--brax_label_info_path', type=str, default='./brax/brax_label_info.npy')
    args = parser.parse_args()
    return args

def get_split_dict(info_dict_path: str, dataset_name: str, seed: int, n_folds: int):
    info_dict = np.load(info_dict_path, allow_pickle=True).item()

    fnames = info_dict['fnames']
    fpaths = np.array([f'/{dataset_name}/images/{x}.png' for x in fnames])
    mask_fpaths = np.array([f'/{dataset_name}/masks/{x}.png' for x in fnames])
    labels = info_dict['labels']

    # u-ones policy
    labels[labels==-1]=1

    split_info_dict = {}
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_number, (train_index, test_index) in enumerate(mskf.split(fpaths, labels)):
        # get test fnames and labels for current fold 
        train_val_fpaths, test_fpaths = fpaths[train_index], fpaths[test_index]
        train_val_labels, test_labels = labels[train_index], labels[test_index]
        train_val_mask_fpaths, test_mask_fpaths = mask_fpaths[train_index], mask_fpaths[test_index]
        
        # get train and val labels for current fold
        mskf2 = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_index, val_index in mskf2.split(train_val_fpaths, train_val_labels):
            train_fpaths, val_fpaths = train_val_fpaths[train_index], train_val_fpaths[val_index]
            train_labels, val_labels = train_val_labels[train_index], train_val_labels[val_index]
            train_mask_fpaths, val_mask_fpaths = train_val_mask_fpaths[train_index], train_val_mask_fpaths[val_index]
            break
        
        # check for duplicates in train, val, and test splits
        setA = set(train_fpaths)
        setB = set(val_fpaths)
        setC = set(test_fpaths)        
        if len(setA.intersection(setC))!=0 or len(setA.intersection(setB))!=0 or len(setB.intersection(setC))!=0:
            raise Exception('Error: Duplicate found in the splits!')
            
        # save split info 
        split_info_dict.update({f'fold_{fold_number}_train_fpaths': train_fpaths})
        split_info_dict.update({f'fold_{fold_number}_train_labels': train_labels})
        split_info_dict.update({f'fold_{fold_number}_train_mask_fpaths': train_mask_fpaths})
        
        split_info_dict.update({f'fold_{fold_number}_val_fpaths': val_fpaths})
        split_info_dict.update({f'fold_{fold_number}_val_labels': val_labels})
        split_info_dict.update({f'fold_{fold_number}_val_mask_fpaths': val_mask_fpaths})
        
        split_info_dict.update({f'fold_{fold_number}_test_fpaths': test_fpaths})
        split_info_dict.update({f'fold_{fold_number}_test_labels': test_labels})
        split_info_dict.update({f'fold_{fold_number}_test_mask_fpaths': test_mask_fpaths})
    
    return split_info_dict

def get_merge_split_dicts(split_dict_1: dict, split_dict_2: dict):
    split_dicts = [split_dict_1, split_dict_2]
    merged_split_dict = {}

    for fold_number in range(5):
        for split_type in ['train', 'val', 'test']:

            fpaths = []
            labels = []
            mask_fpaths = []
            
            for x in split_dicts:
                if len(labels) == 0:
                    fpaths = x[f'fold_{fold_number}_{split_type}_fpaths']
                    labels = x[f'fold_{fold_number}_{split_type}_labels']
                    mask_fpaths = x[f'fold_{fold_number}_{split_type}_mask_fpaths']
                else:
                    fpaths = np.concatenate([fpaths, x[f'fold_{fold_number}_{split_type}_fpaths']])
                    labels = np.concatenate([labels, x[f'fold_{fold_number}_{split_type}_labels']])
                    mask_fpaths = np.concatenate([mask_fpaths, x[f'fold_{fold_number}_{split_type}_mask_fpaths']])
                    
            merged_split_dict.update({
                f'fold_{fold_number}_{split_type}_mask_fpaths': mask_fpaths,
                f'fold_{fold_number}_{split_type}_fpaths': fpaths,
                f'fold_{fold_number}_{split_type}_labels': labels,
                })
    
    return merged_split_dict

def get_test_dict(info_dict_path: str, dataset_name: str):
    info_dict = np.load(info_dict_path, allow_pickle=True).item()
    fnames = info_dict['fnames']
    fpaths = np.array([f'/{dataset_name}/images/{x}.png' for x in fnames])
    mask_fpaths = np.array([f'/{dataset_name}/masks/{x}.png' for x in fnames])
    labels = info_dict['labels']

    # u-ones policy
    labels[labels==-1]=1
    
    test_dict = {}
    test_dict.update({'test_fpaths': fpaths})
    test_dict.update({'test_labels': labels})
    test_dict.update({'test_mask_fpaths': mask_fpaths})
    
    return test_dict
    
def main():
    # get args
    args = get_args()
    
    mimic_split_dict = get_split_dict(args.mimic_label_info_path,'mimic', args.seed, args.n_folds)
    chexpert_split_dict = get_split_dict(args.chexpert_label_info_path,'chexpert', args.seed, args.n_folds)    
    merged_split_dict = get_merge_split_dicts(chexpert_split_dict, mimic_split_dict)
    brax_test_dict = get_test_dict(args.brax_label_info_path, 'brax')
    
    os.makedirs('./split_and_test_dicts/', exist_ok=True)
    np.save('./split_and_test_dicts/mimic_split_info_dict.npy', mimic_split_dict)
    np.save('./split_and_test_dicts/chexpert_split_info_dict.npy', chexpert_split_dict)
    np.save('./split_and_test_dicts/chexpert_mimic_split_info_dict.npy', merged_split_dict)
    np.save('./split_and_test_dicts/brax_test_dict.npy', brax_test_dict)
    
if __name__ == '__main__':
    main()