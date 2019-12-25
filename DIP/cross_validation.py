# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:26:10 2019

@author: DCMC
"""
import sys
import os
import numpy as np
import torch

from eval import eval_net
from unet import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from train import train_net

def make_dataset(img_scale = 0.2):
    train_dataset, val_dataset = [], []
    for i in range(1, 4):
        dir_img = 'data/f0' + str(i) + '/imgs/'
        dir_mask = 'data/f0' + str(i) + '/masks/'
        dir_img_val = 'data/f0' + str(i) + '/imgs_val/'
        dir_mask_val = 'data/f0' + str(i) + '/masks_val/'
        train_dataset.append(BasicDataset(dir_img, dir_mask, img_scale)   )
        val_dataset.append(BasicDataset(dir_img_val, dir_mask_val, img_scale)  )
    
    return train_dataset, val_dataset

def cross_validation():
    train_dataset, val_dataset = make_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    
    val_score = []        
    for i in range(3):
        net = UNet(n_channels=1, n_classes=1) # input R=G=B = gray scale
    
        # get pretrain model
        net.load_state_dict(
                torch.load("./model/PRE_BEST.pth", map_location=device)
                )
        net.to(device=device)
        
        # faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True        
          
        try:
            train_net(net,
                      train_dataset[i],
                      device,
                      epochs=2,
                      batch_size=4,
                      lr=0.01,
                      val_percent=0,
                      save_cp=False,
                      img_scale=0.2, 
                      data_augment=True)
            
        except KeyboardInterrupt:
            torch.save(net.state_dict(), './model/INTERRUPTED.pth')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)   
        
        val_loader = DataLoader(val_dataset[i], batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
        current_score = eval_net(net, val_loader, device, n_val = 20)
        val_score.append(current_score)
    
    print("")
    print(val_score)
    return np.sum(val_score) / 3

if __name__ == '__main__':
    print('Averge score = ' + str(cross_validation()))
    