"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
Some part of the code is adopted from:
https://github.com/arnab39/Semi-supervised-segmentation-cycleGAN
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as tv

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()
    
def get_transformation(size: int):
    transfom_lst = [
        tv.Resize(size),
        tv.ToTensor(),
        tv.Normalize([.5], [.5])
    ]  
    input_transform = tv.Compose(transfom_lst)
    target_transform = tv.Compose([
        tv.Resize(size, interpolation=0),
        ToLabel()
    ])
    transform = {'img': input_transform, 'gt': target_transform}
    return transform

class SegmentationDS(Dataset):
    def __init__(self, img_fpaths: np.array, gt_fpaths: np.array, resize_dim: int = 256):
        super().__init__()
        self.img_fpaths = img_fpaths
        self.gt_fpaths = gt_fpaths
        self.transformation = get_transformation((resize_dim, resize_dim))
       
    def __len__(self):
        return self.img_fpaths.shape[0]

    def __getitem__(self, index):
        img = Image.open(self.img_fpaths[index]).convert('RGB')
        lbl = Image.open(self.gt_fpaths[index])
        img = self.transformation['img'](img)
        lbl = self.transformation['gt'](lbl)
        lbl[lbl==2] = 0
        return img, lbl 
    
class SegmentationDS_TEST(Dataset):
    def __init__(self, test_path: str, resize_dim: int = 256):
        super().__init__()
        self.fnames = np.array(os.listdir(test_path))
        self.img_fpaths = np.array([test_path+x for x in self.fnames])
        self.transformation = get_transformation((resize_dim, resize_dim))
        print(f'Found {self.fnames.shape[0]} test images!')
        
    def __len__(self):
        return self.fnames.shape[0]

    def __getitem__(self, index):
        img = Image.open(self.img_fpaths[index]).convert('RGB')
        img = self.transformation['img'](img)
        return img, self.fnames[index]


# if __name__ == '__main__':
#     jsrt_dir = './datasets/jsrt/'
#     jsrt_gt_dir = './datasets/jsrt_gt/'
#     jsrt_fnames = os.listdir(jsrt_dir)
#     jsrt_fpaths = np.array([jsrt_dir+x for x in jsrt_fnames])
#     jsrt_gt_fpaths = np.array([jsrt_gt_dir+x for x in jsrt_fnames])
    
#     ranzcr_clip_dir = './datasets/ranzcr_clip/'
#     ranzcr_clip_gt_dir = './datasets/ranzcr_clip_gt/'
#     ranzcr_clip_fnames = os.listdir(ranzcr_clip_dir)
#     ranzcr_clip_fpaths = np.array([ranzcr_clip_dir+x for x in ranzcr_clip_fnames])
#     ranzcr_clip_gt_fpaths = np.array([ranzcr_clip_gt_dir+x[:-3]+'png' for x in ranzcr_clip_fnames])
    
#     img_fpaths = ranzcr_clip_fpaths #jsrt_fpaths
#     gt_fpaths = ranzcr_clip_gt_fpaths #jsrt_gt_fpaths
    
#     train_dataset = SegmentationDS(img_fpaths, gt_fpaths)
#     img, gt = train_dataset[250]
#     plt.imshow(gt);plt.show();
    
#     test_dataset = SegmentationDS_TEST('./datasets/jsrt/')
#     img, fname = test_dataset[0]
#     plt.imshow(img[0]);plt.show();
                           