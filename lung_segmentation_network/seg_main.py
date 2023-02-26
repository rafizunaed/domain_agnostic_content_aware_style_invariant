"""
version: 19 February, 2023
Programmed by Mohammad Zunaed
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from seg_trainer_callbacks import set_random_state, AverageMeter
from seg_trainer import ModelTrainer
from seg_dataset import SegmentationDS
from seg_model import define_net
from argparse import ArgumentParser

# import warnings
# warnings.filterwarnings('ignore')

def get_args():
    parser = ArgumentParser(description='lung segmentation network')
    parser.add_argument('--seed', type=int, default=4690)
    parser.add_argument('--image_resize_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weight_saving_path', type=str, default='./weights/segmentation_checkpoint/')
    args = parser.parse_args()
    return args

def get_loaders(args):
    jsrt_dir = './datasets/jsrt/'
    jsrt_gt_dir = './datasets/jsrt_gt/'
    jsrt_fnames = os.listdir(jsrt_dir)
    jsrt_fpaths = np.array([jsrt_dir+x for x in jsrt_fnames])
    jsrt_gt_fpaths = np.array([jsrt_gt_dir+x for x in jsrt_fnames])
    
    ranzcr_clip_dir = './datasets/ranzcr_clip/'
    ranzcr_clip_gt_dir = './datasets/ranzcr_clip_gt/'
    ranzcr_clip_fnames = os.listdir(ranzcr_clip_dir)
    ranzcr_clip_fpaths = np.array([ranzcr_clip_dir+x for x in ranzcr_clip_fnames])
    ranzcr_clip_gt_fpaths = np.array([ranzcr_clip_gt_dir+x[:-3]+'png' for x in ranzcr_clip_fnames])
    
    ranzcr_clip_train_imgs_num = int(0.8 * ranzcr_clip_fpaths.shape[0])    
    train_img_fpaths = np.concatenate([jsrt_fpaths, ranzcr_clip_fpaths[:ranzcr_clip_train_imgs_num]])
    train_gt_fpaths = np.concatenate([jsrt_gt_fpaths, ranzcr_clip_gt_fpaths[:ranzcr_clip_train_imgs_num]])
    val_img_fpaths = ranzcr_clip_fpaths[ranzcr_clip_train_imgs_num:]
    val_gt_fpaths = ranzcr_clip_gt_fpaths[ranzcr_clip_train_imgs_num:]
    
    train_dataset = SegmentationDS(train_img_fpaths, train_gt_fpaths, args.image_resize_dim)
    val_dataset = SegmentationDS(val_img_fpaths, val_gt_fpaths, args.image_resize_dim)     
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
    return train_loader, val_loader

def main():
    # set random states and get args
    args = get_args()
    set_random_state(args.seed) 
    
    # get dataloaders
    train_loader, val_loader = get_loaders(args)
    
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
        
    # define net
    model =  define_net(input_nc=3, output_nc=2, ngf=64, netG='unet_256', norm='instance', use_dropout=True, gpu_ids=args.gpu_ids)
    DEVICE = torch.device(f"cuda:{args.gpu_ids[0]}")
    
    args = { 
        'model': model,
        'Loaders': [train_loader, val_loader],  
        'metrics': {
            'loss': AverageMeter,                     
            'IoU': AverageMeter, 
            },
        'lr': args.lr,
        'epochsTorun': args.epochs,
        'checkpoint_saving_path': args.weight_saving_path,
        'DEVICE': DEVICE,
        'image_resize_dim': args.image_resize_dim,
        }
          
    Trainer = ModelTrainer(**args)
    Trainer.fit() 
    
if __name__ == '__main__':
    main()