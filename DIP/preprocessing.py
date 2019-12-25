# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:30:46 2019

@author: DCMC
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

imgs, masks = [], []

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

for i in range(60) :
    if i < 9 :
        ze = '000'
    elif 9 <= i and i < 99 :
        ze = '00'
    
    img = cv2.imread('data/imgs/' + ze + str(i+1) + '.png', 0)
    mask = cv2.imread('data/masks/' + ze + str(i+1) + '.png', 0)

    img = clahe_hist(img)
    
    cv2.imwrite('data/t/'+ ze + str(i+1) + '.png', img)
    
    imgs.append(img)
    masks.append(mask)
    
