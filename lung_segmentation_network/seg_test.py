"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from seg_trainer_callbacks import set_random_state
from seg_dataset import SegmentationDS_TEST
from seg_model import define_net

from PIL import Image

def get_args():
    parser = ArgumentParser(description='lung segmentation network')
    parser.add_argument('--seed', type=int, default=4690)
    parser.add_argument('--image_resize_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--checkpoint_path', type=str, default='./weights/segmentation_checkpoint/')
    parser.add_argument('--test_path', type=str, default='/home/mhealthclust1/Desktop/codebase_rafi/domain_agnostic_codebase/datasets/mimic/images/')
    parser.add_argument('--save_dir', type=str, default='/home/mhealthclust1/Desktop/codebase_rafi/domain_agnostic_codebase/datasets/mimic/masks/')
    args = parser.parse_args()
    return args

def main():
    # set random states and get args
    args = get_args()
    set_random_state(args.seed) 

    # define dataloader
    test_dataset = SegmentationDS_TEST(args.test_path, args.image_resize_dim)          
    test_loader = DataLoader(
                        test_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=args.n_workers,
                        drop_last=False,
                        pin_memory=True,
                        )
    
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
            
    # define model
    model =  define_net(input_nc=3, output_nc=2, ngf=64, netG='unet_256', norm='instance', use_dropout=True, gpu_ids=args.gpu_ids)
    checkpoint = torch.load(args.checkpoint_path+'/checkpoint_best_IoU.pth')
    model.load_state_dict(checkpoint['Model_state_dict'])
    print('iou score: {:.4f}'.format(checkpoint['val_IoU']))
    del checkpoint
    
    os.makedirs(args.save_dir, exist_ok=True)
    DEVICE = torch.device(f"cuda:{args.gpu_ids[0]}")
    model.eval()                  
    torch.set_grad_enabled(False)
    interp = nn.Upsample(size = (args.image_resize_dim, args.image_resize_dim), mode='bilinear', align_corners=True)
    activation_softmax = nn.Softmax2d()
    with torch.no_grad():
        for itera_no, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(DEVICE) 
          
            with torch.cuda.amp.autocast():
                out = model(images)
                out = interp(out)                                                
                outputs = activation_softmax(out)
              
            pred = outputs.data.max(1)[1].cpu().numpy()
                    
            for i in range(pred.shape[0]):
                lm = pred[i,...].copy()
                lm[lm != 1] = 0
                lm = lm*255
             
                final_mask = lm.astype(np.uint8)
                final_mask = Image.fromarray(final_mask)
                final_mask.save(args.save_dir+image_names[i])
             
            # break

if __name__ == '__main__':
    main()
