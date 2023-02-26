"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2     
import os

def get_train_transforms(img_resize_dim: int):      
    return A.Compose([
            A.Resize(width=img_resize_dim, height=img_resize_dim, p=1.0),             
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

def get_valid_transforms(img_resize_dim: int):      
    return A.Compose([
            A.Resize(width=img_resize_dim, height=img_resize_dim, p=1.0),            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)
    
class ThoraxDS(Dataset):
    def __init__(self, 
                 datasets_root_dir: str, 
                 fpaths: np.array, 
                 mask_fpaths: np.array, 
                 labels: np.array, 
                 transform: A.Compose, 
                 apply_il_srm: bool, 
                 apply_lung_mask_crop: bool,
                 mini_imagenet_root_dir: str,
                 style_source: str):
        super().__init__()
        self.fpaths = np.array([datasets_root_dir+x for x in fpaths])
        self.mask_fpaths = np.array([datasets_root_dir+x for x in mask_fpaths])
        self.labels = np.array(labels)
        self.transform = transform
        self.apply_il_srm = apply_il_srm
        self.apply_lung_mask_crop = apply_lung_mask_crop
        self.style_imgs_fpaths = np.array([mini_imagenet_root_dir+x for x in os.listdir(mini_imagenet_root_dir)])
        
        assert style_source in ['natural', 'cxr']
        self.style_source = style_source
        
    def _get_image_mask_label_(self, index):
        image = Image.open(self.fpaths[index]).convert('RGB')
        image = np.array(image)
        
        mask = Image.open(self.mask_fpaths[index]).convert('RGB')
        mask = np.array(mask)
        mask[mask>0]=255
        
        label = self.labels[index]      
        
        return image, mask, label
            
    def _get_random_style_image_(self):
        if self.style_source == 'natural':
            random_index = np.random.choice(np.arange(self.style_imgs_fpaths.shape[0]))
            image = Image.open(self.style_imgs_fpaths[random_index]).convert('RGB')
        elif self.style_source == 'cxr':
            random_index = np.random.choice(np.arange(self.fpaths.shape[0]))
            image = Image.open(self.fpaths[random_index]).convert('RGB')
            image = np.array(image)
            mask = Image.open(self.mask_fpaths[random_index]).convert('RGB')
            mask = np.array(mask)
            mask[mask>0]=255
            if self.apply_lung_mask_crop:
                image, mask = crop_image_by_lung_mask(image, mask)
                
        image = np.array(image)
        transform = self.transform(image=image)
        image = transform['image']
        return image
   
    def __getitem__(self, index):   
        image, mask, label = self._get_image_mask_label_(index)
        
        # crop by mask
        if self.apply_lung_mask_crop:
            image, mask = crop_image_by_lung_mask(image, mask)
        
        # image and mask transform
        transform = self.transform(image=image)
        image = transform['image']                      
        label = torch.tensor(label).float()
        
        # image level style transfer
        image_no_style_transfer = image.clone()              
        if self.apply_il_srm:       
            x = image.clone()
            y = self._get_random_style_image_()
           
            m1 = torch.mean(x, dim=[1,2], keepdim=True)
            v1 = torch.var(x, dim=[1,2], keepdim=True) 
            x = (x - m1) / (v1 + 1e-7).sqrt()
            
            m2 = torch.mean(y, dim=[1,2], keepdim=True)
            v2 = torch.var(y, dim=[1,2], keepdim=True) 
            y = (y-m2) / (v2+1e-7).sqrt()
                
            mf = m2
            vf = v2        
            x = x * (vf + 1e-7).sqrt() + mf
                         
            image = x
   
        return image, image_no_style_transfer, label
    
    def __len__(self):
        return self.fpaths.shape[0]
    
def crop_image_by_lung_mask(image: np.array, mask: np.array):
    mask = mask.astype(np.float32)
    if mask.ndim > 2:    
        mask = mask[:,:,0]
    mask = Image.fromarray(mask)
    H, W, C = image.shape
    mask = mask.resize([W, H])
    mask = np.array(mask)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    
    mask_arg = np.argwhere(mask==255)
    x = mask_arg[:,0]
    y = mask_arg[:,1]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
  
    w = x_max - x_min
    h = y_max - y_min
    f = 0.025
    x_min = (x_min-w*f).astype(np.int64).clip(min=0)
    x_max = (x_max+w*f).astype(np.int64) 
    y_min = (y_min-h*f).astype(np.int64).clip(min=0)
    y_max = (y_max+h*f).astype(np.int64)
    
    image_crop = image[x_min:x_max,y_min:y_max,:] 
    mask_crop = mask[x_min:x_max,y_min:y_max]
    mask_crop = np.stack([mask_crop, mask_crop, mask_crop], axis=2)
    
    return image_crop, mask_crop

class ThoraxDS_Test(Dataset):
    def __init__(self, 
                 datasets_root_dir: str, 
                 fpaths: np.array, 
                 mask_fpaths: np.array, 
                 labels: np.array, 
                 transform: A.Compose,
                 apply_lung_mask_crop: bool,):
        super().__init__()   
        self.fpaths = np.array([datasets_root_dir+x for x in fpaths])
        self.mask_fpaths = np.array([datasets_root_dir+x for x in mask_fpaths])
        self.labels = np.array(labels)
        self.transform = transform
        self.apply_lung_mask_crop = apply_lung_mask_crop
        
    def _get_image_mask_label_(self, index):
        image = Image.open(self.fpaths[index]).convert('RGB')
        image = np.array(image)
        
        mask = Image.open(self.mask_fpaths[index]).convert('RGB')
        mask = np.array(mask)
        mask[mask>0]=255
        
        label = self.labels[index]      
        
        return image, mask, label
            
    def __getitem__(self, index):   
        image, mask, label = self._get_image_mask_label_(index) 
        
        # crop by mask
        if self.apply_lung_mask_crop:
            image, _ = crop_image_by_lung_mask(image, mask)
        
        # image and mask transform
        transform = self.transform(image=image)
        image = transform['image']                      
        label = torch.tensor(label).float()
             
        return image, label
    
    def __len__(self):
        return self.fpaths.shape[0]