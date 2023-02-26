"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

import os
import torch
from torch import nn

from cls_configs import DEVICE
from cls_trainer_callbacks import MetricStoreBox, ExtraMetricMeter, ProgressBar

def check_if_best_value(current_value: float, previous_best_value: float, metric_name: str='loss', mode: str='min'):
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
class ModelTrainerBaseline():
    def __init__(self, 
                 model: torch.nn.Module, 
                 Loaders: list, 
                 metrics: dict, 
                 fold: int, 
                 lr: float, 
                 epochsTorun: int,
                 checkpoint_saving_path: str,
                 gpu_ids: list,
                 ):
        super().__init__()
                   
        self.metrics = metrics
        self.model = model.to(DEVICE)
        self.trainLoader = Loaders[0]
        self.valLoader = Loaders[1]        
        self.fold = fold
        
        if self.fold != None:
            self.checkpoint_saving_path = checkpoint_saving_path + '/fold' + str(self.fold) + '/'
        else:
            self.checkpoint_saving_path = checkpoint_saving_path + '/'
            self.fold = 0       
        os.makedirs(self.checkpoint_saving_path,exist_ok=True)
        
        self.lr = lr
        self.epochsTorun = epochsTorun       
        
        self.best_loss = 9999
        self.best_auc = -9999
        
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        
    def get_checkpoint(self, val_logs):
        checkpoint_dict = {
            'Epoch': self.current_epoch_no,
            'Model_state_dict': self.model.module.state_dict(),
            }                             
        for key in val_logs.keys():
            checkpoint_dict.update({key: val_logs[key]})
            
        return checkpoint_dict
    
    def perform_validation(self, use_progbar=True):
        self.model.eval()
        torch.set_grad_enabled(False)
        val_info_box = MetricStoreBox(self.metrics)
        extra_metric_box = ExtraMetricMeter()
        if use_progbar:
            val_progbar = ProgressBar(len(self.valLoader), f'(val) Fold {self.fold} Epoch {self.current_epoch_no}/{self.epochsTorun}')        
        for itera_no, data in enumerate(self.valLoader):                        
            images, targets = data
            images = images.to(DEVICE) 
            targets = targets.to(DEVICE)                   
            
            with torch.no_grad() and torch.cuda.amp.autocast(): 
                out = self.model(images)                                 
                batch_loss = self.criterion_bce(out['logits'], targets)
                        
            # update extra metric      
            y_score = out['logits'].detach().cpu().clone().float().sigmoid().numpy()
            y_true = targets.detach().cpu().data.numpy()
            extra_metric_box.update(y_score, y_true)
            
            # update progress bar, info box
            val_info_box.update({'loss':[batch_loss.detach().item(), targets.shape[0]]})
            logs_to_display=val_info_box.get_value()
            logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            if use_progbar:
                val_progbar.update(1, logs_to_display)
        
        # calculate all metrics and close progbar
        logs_to_display=val_info_box.get_value()
        auc = extra_metric_box.feedback()
        logs_to_display.update({'auc': auc})
        logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
        
        val_logs = logs_to_display
        self.best_loss, is_best_loss = check_if_best_value(val_logs['val_loss'], self.best_loss, 'loss', 'min')
        self.best_auc, is_best_auc = check_if_best_value(val_logs['val_auc'], self.best_auc, 'auc', 'max')
        
        checkpoint_dict = self.get_checkpoint(val_logs)                              
        if is_best_auc:                                    
            torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_auc_fold{}.pth'.format(self.fold))       
        del checkpoint_dict
        
        best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
        logs_to_display.update(best_results_logs)
        if use_progbar:
            val_progbar.update(logs_to_display=logs_to_display)
            val_progbar.close() 
        
    def train_one_epoch(self):
        train_info_box = MetricStoreBox(self.metrics)
        extra_metric_box = ExtraMetricMeter()
        train_progbar = ProgressBar(len(self.trainLoader), f'(Train) Fold {self.fold} Epoch {self.current_epoch_no}/{self.epochsTorun}')
        for itera_no, data in enumerate(self.trainLoader):                                   
            self.model.train()
            torch.set_grad_enabled(True) 
            self.optimizer.zero_grad()
              
            images, _, targets = data
            images = images.to(DEVICE) 
            targets = targets.to(DEVICE)                
            
            with torch.cuda.amp.autocast():
                out = self.model(images, True, targets)  
                cls_loss = out['cls_loss'].mean()
                batch_loss = cls_loss
                    
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer) 
            self.scaler.update()  
            self.optimizer.zero_grad()
                   
            # update extra metric
            y_score = out['logits'].detach().cpu().clone().float().sigmoid().numpy()
            y_true = targets.detach().cpu().data.numpy()
            extra_metric_box.update(y_score, y_true)
            
            # update progress bar, info box
            train_info_box.update({'loss':[batch_loss.detach().item(), targets.shape[0]],
                                   'cls_loss':[cls_loss.detach().item(), targets.shape[0]],
                                   })
            logs_to_display=train_info_box.get_value()
            logs_to_display = {f'train_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
            logs_to_display.update(best_results_logs)
            train_progbar.update(1, logs_to_display)
            
        # calculate all metrics and close progbar
        logs_to_display=train_info_box.get_value()
        best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
        logs_to_display.update(best_results_logs)
        auc = extra_metric_box.feedback()
        logs_to_display.update({'auc': auc})
        logs_to_display = {f'train_{key}': logs_to_display[key] for key in logs_to_display.keys()}
        train_progbar.update(logs_to_display=logs_to_display)
        train_progbar.close()
        
#%% train part starts here
    def fit(self):   
        for epoch in range(self.epochsTorun):
            self.current_epoch_no = epoch+1
            self.train_one_epoch()
            self.perform_validation()