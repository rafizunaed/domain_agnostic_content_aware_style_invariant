"""
version: 18 February, 2023
Process MIMIC-CXR-JPG Dataset
https://physionet.org/content/mimic-cxr-jpg/2.0.0/
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
    parser = ArgumentParser(description='process mimic-cxr-jpg')
    parser.add_argument('--mimic_labeler_csv_path', type=str, default='/media/mhealthclust1/data/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv')
    parser.add_argument('--mimic_metadata_csv_path', type=str, default='/media/mhealthclust1/data/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
    parser.add_argument('--mimic_data_path', type=str, default='/media/mhealthclust1/data/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/')
    parser.add_argument('--image_resize_dim', type=int, default=512)
    parser.add_argument('--save_root_dir', type=str, default='./mimic/')   
    args = parser.parse_args()
    return args

def main():
    # set random states and get args
    set_random_state(4690)  
    args = get_args()
    
    # read mimic labeler and metadata
    df = pd.read_csv(args.mimic_labeler_csv_path)
    df = df.fillna(0)    
    df_metadata = pd.read_csv(args.mimic_metadata_csv_path)
    
    keys_to_labels = {}
    for i in tqdm(range(df.shape[0]), desc='reading mimic labeler csv'):
        key = str(int(df.iloc[i].subject_id))+'_'+str(int(df.iloc[i].study_id))
        label = [df.iloc[i][x] for x in pathology_names]
        keys_to_labels.update({key:np.array(label)})
        # break
       
    # get patient id, img_path, and label
    pat_id = []
    img_path = []
    all_keys = []
    for i in tqdm(range(df_metadata.shape[0]), desc='preparing mimic info'):
        # we only use frontal images
        if df_metadata.iloc[i]['ViewPosition'] in ['PA', 'AP']:
            pat_id.append(int(df_metadata.iloc[i].subject_id))
            key = str(int(df_metadata.iloc[i].subject_id))+'_'+str(int(df_metadata.iloc[i].study_id))
            all_keys.append(key)
            
            p = str(int(df_metadata.iloc[i].subject_id))
            s = str(int(df_metadata.iloc[i].study_id))
            
            fpath = args.mimic_data_path + f'/p{p[:2]}/p{p}/s{s}/' + df_metadata.iloc[i].dicom_id + '.jpg'
            img_path.append(fpath)            
        # break      
    pat_id = np.array(pat_id)
    img_path = np.array(img_path)
    all_keys = np.array(all_keys)
    
    # we select one image per patient
    os.makedirs(args.save_root_dir+'/images/', exist_ok=True)
    _, unique_indexes = np.unique(pat_id, return_index=True)
    label_info = {'fnames': [], 'labels': []}
    for idx in tqdm(unique_indexes, total=unique_indexes.shape[0], desc='processing mimic images'):
        key = all_keys[idx]
        if key in keys_to_labels.keys():       
            fp = img_path[idx]
            img = Image.open(fp)
            img = img.resize([args.image_resize_dim, args.image_resize_dim])
            new_fname = ('_').join(fp.split('/')[-4:])[:-4]
            img.save(f'{args.save_root_dir}/images/{new_fname}.png')
            
            label = keys_to_labels[key]
            label_info['fnames'].append(new_fname)
            label_info['labels'].append(label)
        
        # break
    
    label_info['fnames'] = np.array(label_info['fnames'])
    label_info['labels'] = np.array(label_info['labels'])
    np.save(f'{args.save_root_dir}/mimic_label_info.npy', label_info)
    
if __name__ == '__main__':
    main()