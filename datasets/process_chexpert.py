"""
version: 19 February, 2023
Process CheXpert Dataset
https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
Programmed by Mohammad Zunaed
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

pathology_names = ['Atelectasis', 'Cardiomegaly', 'Enlarged Cardiomediastinum', 'Consolidation', 'Edema', 'Pneumonia', 'Pneumothorax',
                   'Pleural Effusion', 'Pleural Other', 'Lung Lesion', 'Lung Opacity', 'Fracture', 'Support Devices', 'No Finding']

def set_random_state(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    
def get_args():
    parser = ArgumentParser(description='process cheXpert')
    parser.add_argument('--chexpert_frontal_csv_path', type=str, default='./cheXpert_train_frontal.csv')
    parser.add_argument('--chexpert_data_path', type=str, default='/media/mhealthclust1/data/chexpert full/CheXpert-v1.0/')
    parser.add_argument('--image_resize_dim', type=int, default=512)
    parser.add_argument('--save_root_dir', type=str, default='./chexpert/')   
    args = parser.parse_args()
    return args

def main():
    # set random states and get args
    set_random_state(4690)  
    args = get_args()
    
    # read chexpert train frontal csv
    df = pd.read_csv(args.chexpert_frontal_csv_path)
    df = df.fillna(0)
    
    # get patient id, img_path, and label
    pat_id = []
    img_path = []
    dis_label = []   
    for i in tqdm(range(df.shape[0]), desc='reading cheXpert frontal csv'):
        pat_id.append(df.iloc[i].Path.split('/')[2])
        img_path.append(df.iloc[i].Path)
        label = [df.iloc[i][x] for x in pathology_names]
        dis_label.append(label)
        # break
    dis_label = np.array(dis_label)
    pat_id = np.array(pat_id)
    img_path = np.array(img_path)
    
    # we select one image per patient
    os.makedirs(args.save_root_dir+'/images/', exist_ok=True)
    _, unique_indexes = np.unique(pat_id, return_index=True)
    label_info = {'fnames': [], 'labels': []}
    for idx in tqdm(unique_indexes, total=unique_indexes.shape[0], desc='processing cheXpert images'):
        fp = '/'.join(img_path[idx].split('/')[1:])
        img = Image.open(f'{args.chexpert_data_path}/{fp}')
        img = img.resize([args.image_resize_dim, args.image_resize_dim])
        new_fname = '_'.join(fp.split('/')[1:])[:-4]
        img.save(f'{args.save_root_dir}/images/{new_fname}.png')      
        
        label = dis_label[idx]
        label_info['fnames'].append(new_fname)
        label_info['labels'].append(label)
        # break
    
    label_info['fnames'] = np.array(label_info['fnames'])
    label_info['labels'] = np.array(label_info['labels'])
    np.save(f'{args.save_root_dir}/chexpert_label_info.npy', label_info)
    
if __name__ == '__main__':
    main()