# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:46:46 2019

@author: DCMC
"""
import os
import sys
import logging
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/generate_imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5, 
              data_augment=True):
    
    dataset = BasicDataset(dir_img, dir_mask, img_scale)      
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    global_step = 0
    
    logging.info(f'''Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {lr}
    Training size:   {n_train}
    Validation size: {n_val}
    Checkpoints:     {save_cp}
    Device:          {device.type}
    Images scaling:  {img_scale}
    ''')
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss() # 1 class
    best_score = 0.
    
    for epoch in range(epochs):
        net.train()
        
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert true_masks.shape[1] == net.n_classes, \
                    f'Network has been defined with {net.n_classes} output classes, ' \
                    f'but loaded masks have {true_masks.shape[1]} channels. Please check that ' \
                    'the masks are loaded correctly.'
                
                if data_augment:
                    for i in range(imgs.__len__()):
                        imgs[i], true_masks[i] = my_segmentation_transforms(imgs[i], true_masks[i])
                
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    print(" ")
                    print('Validation Dice Coeff: {}'.format(val_score))
        
        if best_score < val_score:
            torch.save(net.state_dict(), 'BEST.pth')
            logging.info(f'Best saved !')
            best_score = val_score
                              
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        
def my_segmentation_transforms(image, segmentation):
    
    image = image.numpy().transpose((1, 2, 0)) * 255
    segmentation = segmentation.numpy().transpose((1, 2, 0)) * 255
    
    image, segmentation = TF.to_pil_image(image.astype(np.float32)),\
                        TF.to_pil_image(segmentation.astype(np.float32))
    
    if random.random() > 0.5:
        image = TF.affine(image, 15, (2, 2), 1, 0.2)
        segmentation = TF.affine(segmentation, 15, (2, 2), 1, 0.2)       
        
    if random.random() > 0.5:
        image = TF.hflip(image)
        segmentation = TF.hflip(segmentation)
    
    image, segmentation = TF.to_tensor(image) / 255, TF.to_tensor(segmentation) / 255
    
    return image, segmentation

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=256,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1) # input R=G=B = gray scale
    
    # for pre-train
    args.load = ""
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
    
    net.to(device=device)
    # faster convolutions, but more memory
    torch.backends.cudnn.benchmark = True
    
try:
    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              img_scale=args.scale,
              val_percent=args.val / 100,
              data_augment=True)
    
except KeyboardInterrupt:
    torch.save(net.state_dict(), 'INTERRUPTED.pth')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

