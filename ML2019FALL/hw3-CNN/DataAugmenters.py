import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

class ImageGenerator():
    def __init__(self, imgs, label): # input tensor
        self.imgs = imgs
        self.label = label
        self.augs = []
        self.augs_label = []
        
    def _imshow(self, index):
        npimg = self.imgs.numpy()[index] # size(1,48,48)
        npimg = npimg[0] # size(48,48)
        plt.imshow(npimg) 
       
    def _normalize(self,method='ZoomOut',mean_or_max='None',std_or_min='None',trans=False):
        if trans is False:
            if method == 'Rescaling':
                mean_or_max = torch.max(self.imgs, dim = 0) # (Number of Images, RGB, height, width)
                std_or_min = torch.min(self.imgs, dim = 0)  
            elif method == 'Standardization':
                mean_or_max = torch.mean(self.imgs, dim = 0)
                std_or_min = torch.std(self.imgs, dim = 0)          
        if method == 'Rescaling':
            self.imgs = (self.imgs - std_or_min) / (mean_or_max - std_or_min)
        elif method == 'Standardization':
            self.imgs = (self.imgs - mean_or_max) / std_or_min
        elif method == 'ZoomOut':
            self.imgs /= 255.0
            return
        if trans is False:
            return mean_or_max, std_or_min

    def _horizontal_flip(self):
        for i in range(len(self.imgs)):
            if np.random.rand() < 0.5:
                self.augs += list(flip(self.imgs[i], 2))
                self.augs_label += [self.label[i]]
    
    def _vertical_flip(self):
        for i in range(len(self.imgs)):
            if np.random.rand() < 0.5:
                self.augs += list(flip(self.imgs[i], 0))
                self.augs_label += [self.label[i]]            

    def _rotate_flip(self):
        for i in range(len(self.imgs)):
            if np.random.rand() < 0.5:
                self.augs += list(flip(self.imgs[i], 1))
                self.augs_label += [self.label[i]]
        for i in range(len(self.imgs)):
            if np.random.rand() < 0.5:
                self.augs += list(flip(flip(self.imgs[i], 1), 2))
                self.augs_label += [self.label[i]]
   
    def _rand_earsing(self):
        for i in range(len(self.imgs)):
            if np.random.rand() < 0.5:                
                self.augs += list(transforms.RandomErasing(p=1)(self.imgs[i]))
                self.augs_label += [self.label[i]]
    
    def _make_augment_images(self):
        img_aug = torch.FloatTensor(len(self.augs), 1, len(self.augs[0]), len(self.augs[1]))
        label_aug = torch.LongTensor(np.array(self.augs_label))
        for i in range(len(self.augs)):
            img_aug[i] = self.augs[i]
        self.imgs = torch.cat((self.imgs, img_aug)) 
        self.label = torch.cat((self.label, label_aug))
        