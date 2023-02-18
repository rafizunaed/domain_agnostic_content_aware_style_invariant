"""
version: 18 February, 2023
Process BRAX Dataset
https://physionet.org/content/brax/1.1.0/
Programmed by Mohammad Zunaed
"""

import os
import random
import pandas as pd
import numpy as np
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
    parser = ArgumentParser(description='process brax')
    parser.add_argument('--brax_csv_path', type=str, default='/media/mhealthclust1/data/brax/physionet.org/files/brax/1.1.0/master_spreadsheet_update.csv')
    parser.add_argument('--brax_data_path', type=str, default='/media/mhealthclust1/data/brax/physionet.org/files/brax/1.1.0/')
    parser.add_argument('--image_resize_dim', type=int, default=512)
    parser.add_argument('--save_root_dir', type=str, default='./brax/')   
    args = parser.parse_args()
    return args

def main():
    # set random states and get args
    set_random_state(4690)  
    args = get_args()
    
    # read brax csv
    df = pd.read_csv(args.brax_csv_path)
    df = df.fillna(0)
    
    # get patient id, img_path, and label
    pat_id = []
    img_path = []
    dis_label = []
    for i in tqdm(range(df.shape[0]), desc='reading brax csv'):
        # we only use frontal images
        if df.iloc[i]['ViewPosition'] in ['PA', 'AP']:
            pat_id.append(df.iloc[i]['PatientID'])
            img_path.append(df.iloc[i]['PngPath'])           
            label = [df.iloc[i][x] for x in pathology_names]
            dis_label.append(label)
    dis_label = np.array(dis_label)
    pat_id = np.array(pat_id)
    img_path = np.array(img_path)
    
    # we select one image per patient
    os.makedirs(args.save_root_dir+'/images/', exist_ok=True)
    _, unique_indexes = np.unique(pat_id, return_index=True)
    label_info = {'fnames': [], 'labels': []}
    for idx in tqdm(unique_indexes, total=unique_indexes.shape[0], desc='processing brax images'):
        fp = img_path[idx]
        fp = fp[:-7]
        img = Image.open(f'{args.brax_data_path}/{fp}')
        img = img.resize([args.image_resize_dim, args.image_resize_dim])
        new_fname = fp.split('/')[-1]
        img.save(f'{args.save_root_dir}/images/{new_fname}.png')
        
        label = dis_label[idx]
        label_info['fnames'].append(new_fname)
        label_info['labels'].append(label)
        
        # break
    
    label_info['fnames'] = np.array(label_info['fnames'])
    label_info['labels'] = np.array(label_info['labels'])
    np.save(f'{args.save_root_dir}/brax_label_info.npy', label_info)
    

if __name__ == '__main__':
    main()