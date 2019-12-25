# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 20:01:34 2019

@author: DCMC
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch

from PIL import Image
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

img = cv.imread('./predict/img.png')
mask = cv.imread('./predict/mask.png')
predict = cv.imread('./predict/predict.png')

def aliasing(mask, predict) :
    if mask.shape[0] / predict.shape[0] == mask.shape[1] / predict.shape[1] :
        scale = mask.shape[0] / predict.shape[0]
    h, w = predict.shape[0], predict.shape[1]
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
   
    return cv.resize(predict, (newW, newH), cv.INTER_AREA)

predict = aliasing(mask, predict)


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img_gray, 127, 255, 0)

image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

plt.imshow(predict)

for contour in contours:
    cv.drawContours(image, contour, -1, (255, 255, 255), 3)
    
cv.drawContours(image, contours, -1, (0, 0, 255), 3)
cv.namedWindow('Image With Contours', cv.WINDOW_NORMAL)
cv.imshow('Image With Contours', image)


