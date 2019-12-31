# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:23:28 2019

@author: DCMC
"""

import sys
import os
import numpy as np
import torch

from predict import *
from eval import eval_net
from resunet import ResidualUNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from PIL import Image
from train import train_net

dir_img = 'data/total/imgs_train/'
dir_mask = 'data/total/masks/'


def ensembleTrain(img_scale=0.2, N=8):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    train_dataset = BasicDataset(dir_img, dir_mask, img_scale)    
    
    for i in range(N):
        net = ResidualUNet(n_channels=1, n_classes=1) # input R=G=B = gray scale        
        # net.load_state_dict(
        #     torch.load('model/PRE_BEST.pth', map_location=device)
        # )
        net.to(device=device)
        # faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True        
          
        try:
            train_net(net,
                      train_dataset,
                      device,
                      epochs=512,
                      batch_size=4,
                      lr=0.001,
                      val_percent=0,
                      save_cp=False,
                      img_scale=img_scale, 
                      data_augment=True)
            
        except KeyboardInterrupt:
            torch.save(net.state_dict(), './model/INTERRUPTED.pth')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)   
        
        torch.save(net.state_dict(), './model/BEST_' + str(i+1) +'.pth')
    
    return

def ensembleEval(img_scale=0.2, N=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = BasicDataset(dir_imgs_val, dir_masks_val, img_scale)
    
    val_score = []
    for i in range(N):
        net = ResidualUNet(n_channels=1, n_classes=1) # input R=G=B = gray scale
        net.to(device=device)
        
        Loaded_model = './model/BEST_' + str(i+1) + '.pth'
        
        net.load_state_dict(torch.load(Loaded_model, map_location=device))
        print('load model<<' + Loaded_model)
        
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,\
                                num_workers=0, pin_memory=True)
        current_score = eval_net(net, val_loader, device, n_val = 20)
        val_score.append(current_score)
    
    print('')
    print(val_score)
    return np.sum(val_score) / N

def ensemblePredict(img, scale_factor=0.2, out_threshold=0.9,\
                    mean_thre=0.9, N=8):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    masks = []
    for i in range(N):
        net = ResidualUNet(n_channels=1, n_classes=1) # input R=G=B = gray scale
        net.to(device=device)
        
        Loaded_model = './model/BEST_' + str(i+1) + '.pth'
        
        net.load_state_dict(torch.load(Loaded_model, map_location=device))
        print('load model<<' + Loaded_model)
    
        mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=scale_factor,
                            out_threshold=out_threshold,
                            device=device)
        
        masks.append(mask)
    
    mask_mean = np.mean(masks, axis=0)
    where_is_mask = mask_mean > mean_thre
    
    return where_is_mask


if __name__ == '__main__':
    in_files = "data/total/imgs_train/0058.png"
    img = Image.open(in_files)
    
    N = 8
    ## Train
    # ensembleTrain(N=N)
    
    ## Eval
    # print('avg_score=' + str(ensembleEval(N=N)))
    
    ## Predict
    print('predict')
    predict = ensemblePredict(img, out_threshold=0.99, N=N)
    plot_img_and_mask(img, predict)
    
    # write
    mask_bgr = mask_to_image(predict)    
    mask_bgr = mask_bgr.resize((500, 1200))
    mask_bgr.save('./predict/predict.png')
