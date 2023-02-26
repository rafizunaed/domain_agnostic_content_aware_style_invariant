"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from seg_trainer_callbacks import ProgressBar, MetricStoreBox

def check_if_best_value(current_value, previous_best_value, metric_name='loss', mode='min'):
    if mode == 'min':
        if previous_best_value > current_value:
            print('\033[32;1m' + ' Val {} is improved from {:.4f} to {:.4f}! '.format(metric_name, previous_best_value, current_value) + '\033[0m')
            best_value = current_value
            is_best_value = True
        else:
            print('\033[31;1m' + ' Val {} is not improved from {:.4f}! '.format(metric_name, previous_best_value) + '\033[0m')
            best_value = previous_best_value
            is_best_value = False
    else:
        if previous_best_value < current_value:
            print('\033[32;1m' + ' Val {} is improved from {:.4f} to {:.4f}! '.format(metric_name, previous_best_value, current_value) + '\033[0m')
            best_value = current_value
            is_best_value = True
        else:
            print('\033[31;1m' + ' Val {} is not improved from {:.4f}! '.format(metric_name, previous_best_value) + '\033[0m')
            best_value = previous_best_value
            is_best_value = False
            
    return best_value, is_best_value
      
#%% #################################### Model Trainer Class #################################### 
class ModelTrainer():
    def __init__(self, 
                 model=None, 
                 Loaders=[None,[]], 
                 metrics=None, 
                 lr=None, 
                 epochsTorun=None,
                 checkpoint_saving_path=None,                
                 DEVICE=None,
                 image_resize_dim=None,
                 ):     
        super().__init__()
                   
        self.metrics = metrics
        self.model = model.to(DEVICE)
        self.trainLoader = Loaders[0]
        self.valLoader = Loaders[1]        
        
        self.checkpoint_saving_path = checkpoint_saving_path + '/'       
        os.makedirs(self.checkpoint_saving_path,exist_ok=True)
        
        self.lr = lr
        self.epochsTorun = epochsTorun   
        self.DEVICE = DEVICE
        
        self.best_loss = 9999
        self.best_iou = -9999
               
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.interp = nn.Upsample(size = (image_resize_dim, image_resize_dim), mode='bilinear', align_corners=True)
        self.activation_softmax = nn.Softmax2d()
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, mode='min', patience=5, verbose=True)
        self.scaler = torch.cuda.amp.GradScaler()
             
    def get_checkpoint(self, val_logs):
        checkpoint_dict = {
            'Epoch': self.current_epoch_no,
            'Model_state_dict': self.model.state_dict(),
            }                             
        for key in val_logs.keys():
            checkpoint_dict.update({key: val_logs[key]})
            
        return checkpoint_dict
    
    def perform_validation(self, use_progbar=True):
        self.model.eval()
        torch.set_grad_enabled(False)       
        val_info_box = MetricStoreBox(self.metrics)
        if use_progbar:
            val_progbar = ProgressBar(len(self.valLoader), f'(val) Epoch {self.current_epoch_no}/{self.epochsTorun}')        
        for itera_no, data in enumerate(self.valLoader):                        
            images, targets = data
            images = images.to(self.DEVICE) 
            targets = targets.to(self.DEVICE)                   
            
            with torch.no_grad() and torch.cuda.amp.autocast():               
                out = self.model(images)                                 
                batch_loss = self.criterion_ce(out, targets)                                 
                outputs = self.activation_softmax(out)
                        
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = targets.data.cpu().numpy()            
            batch_lung_iou = 0           
            for i in range(pred.shape[0]):
                lung_gt = gt[i,:,:] == 1                
                lung_pred = pred[i,:,:] == 1               
                batch_lung_iou += np.logical_and(lung_gt, lung_pred).sum()/np.logical_or(lung_gt, lung_pred).sum()
            batch_lung_iou /= pred.shape[0]
            iou = batch_lung_iou  
            
            # update progress bar, info box
            val_info_box.update({'loss':[batch_loss.detach().item(), targets.shape[0]],
                                 'IoU':[iou, targets.shape[0]]})
            logs_to_display=val_info_box.get_value()
            logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            if use_progbar:
                val_progbar.update(1, logs_to_display)
        
        # calculate all metrics and close progbar
        logs_to_display=val_info_box.get_value()
        logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
        
        val_logs = logs_to_display
        self.best_loss, is_best_loss = check_if_best_value(val_logs['val_loss'], self.best_loss, 'loss', 'min')
        self.best_iou, is_best_iou = check_if_best_value(val_logs['val_IoU'], self.best_iou, 'IoU', 'max')
                 
        checkpoint_dict = self.get_checkpoint(val_logs)
                              
        if is_best_iou:                                    
            torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_IoU.pth')
         
        del checkpoint_dict
        
        best_results_logs = {'best_val_loss': self.best_loss, 'best_val_iou':self.best_iou}
        logs_to_display.update(best_results_logs)
        if use_progbar:
            val_progbar.update(logs_to_display=logs_to_display)
            val_progbar.close() 
        
        return val_logs
    
    def train_one_epoch(self):
        train_info_box = MetricStoreBox(self.metrics)
        train_progbar = ProgressBar(len(self.trainLoader), f'(Train) Epoch {self.current_epoch_no}/{self.epochsTorun}')
        for itera_no, data in enumerate(self.trainLoader):                                   
            self.model.train()
            torch.set_grad_enabled(True) 
            self.optimizer.zero_grad()   
            
            images, targets = data
            images = images.to(self.DEVICE) 
            targets = targets.to(self.DEVICE)                            
            
            with torch.cuda.amp.autocast():
                out = self.model(images)
                out = self.interp(out)
                batch_loss = self.criterion_ce(out, targets)
                    
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer) 
            self.scaler.update()    
            self.optimizer.zero_grad()
                   
            # update progress bar, info box
            train_info_box.update({'loss':[batch_loss.detach().item(), targets.shape[0]]})
            logs_to_display=train_info_box.get_value()
            logs_to_display = {f'train_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            best_results_logs = {'best_val_loss': self.best_loss, 'best_val_iou':self.best_iou}
            logs_to_display.update(best_results_logs)
            train_progbar.update(1, logs_to_display)
            
            # break
            
        train_progbar.close()
        

#%% train part starts here
    def fit(self):   
        for epoch in range(self.epochsTorun):
            self.current_epoch_no = epoch+1
            
            self.train_one_epoch()
            val_logs = self.perform_validation()
            
            self.scheduler.step(val_logs['val_loss'])