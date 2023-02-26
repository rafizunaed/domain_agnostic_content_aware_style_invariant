"""
Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

import torch
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F

from cls_configs import DEVICE
from cls_densenet_ibn import densenet121_ibn_a

class DenseNet121_IBN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # get densenet architecture
        model = densenet121_ibn_a(pretrained=True)
        modules = list(model.children())[0]

        # stem
        self.stem = nn.Sequential(*modules[:4])

        # dense block 1
        self.db1 = nn.Sequential(*modules[4:5])
        self.trn1 = nn.Sequential(*list(modules[5])[:-1])

        # dense block 2
        self.db2 = nn.Sequential(*modules[6:7])
        self.trn2 = nn.Sequential(*list(modules[7])[:-1])

        # dense block 3
        self.db3 = nn.Sequential(*modules[8:9])
        self.trn3 = nn.Sequential(*list(modules[9])[:-1])

        # dense block 4
        self.db4 = nn.Sequential(*modules[10:])

        # avg layer for applying after dense block-1,2,3
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # final fc
        self.classifier = nn.Linear(1024, num_classes)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()

    @autocast()
    def get_loss(self, logits, targets):
        criterion_bce = nn.BCEWithLogitsLoss()              
        cls_loss = criterion_bce(logits, targets)
        return cls_loss

    @autocast()
    def forward(self, x, is_training=False, targets=None):
        x = self.stem(x)

        x = self.db1(x)
        x = self.trn1(x)
        x = self.avg(x)

        x = self.db2(x)
        x = self.trn2(x)
        x = self.avg(x)

        x = self.db3(x)
        x = self.trn3(x)
        x = self.avg(x)

        x = self.db4(x)
        xg = F.relu(x)

        xg_pool = F.adaptive_avg_pool2d(xg, (1,1)).flatten(1)       
        logits = self.classifier(xg_pool)

        if is_training:
            cls_loss = self.get_loss(logits, targets)
            return {
                'logits': logits,
                'gfm': xg,
                'gfm_pool': xg_pool,
                'cls_loss': cls_loss,
                } 
        else:
            return {
                'logits': logits,
                'gfm': xg,
                'gfm_pool': xg_pool,
                } 

class DenseNet121_IBN_proposed(nn.Module):
    def __init__(self, num_classes: int, apply_fl_srm: bool=True, apply_l_ccr: bool=True, apply_l_scr: bool=True, apply_gfe_loss: bool=True, fl_srm_level: int=2):
        super().__init__()
        # get densenet architecture
        model = densenet121_ibn_a(pretrained=True)                   
        modules = list(model.children())[0]

        # stem
        self.stem = nn.Sequential(*modules[:4])

        # dense block 1
        self.db1 = nn.Sequential(*modules[4:5])
        self.trn1 = nn.Sequential(*list(modules[5])[:-1])

        # dense block 2
        self.db2 = nn.Sequential(*modules[6:7])
        self.trn2 = nn.Sequential(*list(modules[7])[:-1])

        # dense block 3
        self.db3 = nn.Sequential(*modules[8:9])
        self.trn3 = nn.Sequential(*list(modules[9])[:-1])

        # dense block 4
        self.db4 = nn.Sequential(*modules[10:])

        # avg layer for applying after dense block-1,2,3
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # # final fc
        self.classifier = nn.Linear(1024, num_classes)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()

        self.srm = SRM()

        self.apply_fl_srm = apply_fl_srm
        self.apply_l_ccr = apply_l_ccr
        self.apply_l_scr = apply_l_scr
        self.apply_gfe_loss = apply_gfe_loss
        self.fl_srm_level = fl_srm_level
        
    @autocast()
    def get_cls_loss(self, logits, targets):
        criterion_bce = nn.BCEWithLogitsLoss()             
        cls_loss = criterion_bce(logits, targets)
        return cls_loss

    @autocast()
    def forward(self, x: torch.Tensor, is_training: bool=False, targets=None, y: torch.Tensor=None):
        x = self.stem(x)

        x = self.db1(x)
        x = self.trn1(x)
        x = self.avg(x)

        if is_training and self.apply_fl_srm and self.fl_srm_level==1:
            x = self.srm(x)

        x = self.db2(x)
        x = self.trn2(x)
        x = self.avg(x)

        if is_training and self.apply_fl_srm and self.fl_srm_level==2:
            x = self.srm(x)

        x = self.db3(x)
        x = self.trn3(x)
        x = self.avg(x)

        if is_training and self.apply_fl_srm and self.fl_srm_level==3:
            x = self.srm(x)
                
        x = self.db4(x)
        xg = F.relu(x)

        xg_pool = F.adaptive_avg_pool2d(xg, (1,1)).flatten(1)
        logits = self.classifier(xg_pool)

        if is_training:
            cls_loss = self.get_cls_loss(logits, targets)

            if self.apply_l_ccr or self.apply_l_scr:
                y = self.stem(y)

                y = self.db1(y)
                y = self.trn1(y)
                y = self.avg(y)

                y = self.db2(y)
                y = self.trn2(y)
                y = self.avg(y)

                y = self.db3(y)
                y = self.trn3(y)
                y = self.avg(y)

                y = self.db4(y)
                yg = F.relu(y)

            if self.apply_l_ccr and self.apply_l_scr:
                gm_x = generate_gram_matrix(xg)
                gm_y = generate_gram_matrix(yg)
                consistency_loss = F.mse_loss(xg, yg)+F.mse_loss(gm_x, gm_y) 
            elif self.apply_l_ccr:
                consistency_loss = F.mse_loss(xg, yg)
            elif self.apply_l_scr:
                gm_x = generate_gram_matrix(xg)
                gm_y = generate_gram_matrix(yg)
                consistency_loss = F.mse_loss(gm_x, gm_y)
            else:
                consistency_loss = torch.tensor([0.0]).cuda()

            if self.apply_gfe_loss:
                gfe_loss = get_gfe_loss(xg)
            else:
                gfe_loss = torch.tensor([0.0]).cuda()
 
            return {
                'logits': logits,
                'gfm': xg,
                'gfm_pool': xg_pool,
                'cls_loss': cls_loss,
                'consistency_loss': consistency_loss,
                'gfe_loss': gfe_loss,
                } 
        else:
            return {
                'logits': logits,
                'gfm': xg,
                'gfm_pool': xg_pool,
                }

def generate_gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features = F.normalize(features, dim=2, eps=1e-7)
    features_t = features.transpose(1, 2)
    gram_matrix = features.bmm(features_t)
    return gram_matrix

def get_gfe_loss(x):
    gfm = x.clone()
    criterion_mse = torch.nn.MSELoss()
    gm = generate_gram_matrix(gfm)  
    scores = torch.diagonal(gm, offset=0, dim1=-2, dim2=-1)
    gt = torch.ones_like(scores)
    gfe_loss = criterion_mse(scores, gt)
    return gfe_loss

class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.size()

        # normalize
        x = x.view(N, C, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)      
        x = (x - mean) / (var + self.eps).sqrt()

        # swap styles
        idx_swap = torch.arange(N).flip(0)
        mean = mean[idx_swap]
        var = var[idx_swap]

        x = x * (var + self.eps).sqrt() + mean
        x = x.view(N, C, H, W)

        return x