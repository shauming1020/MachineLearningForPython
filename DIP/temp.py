# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
 
imgl = cv2.imread("0001.png")  #左圖
imgr = cv2.imread("0023.png")   #右圖

gray_l = cv2.cvtColor(imgl,cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(imgr,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray_l,127,255,cv2.THRESH_BINARY)
 
#空圖
img_space1 = np.zeros((1200,500,3), np.uint8) 
img_space1.fill(0)
img_space2 = np.zeros((1200,500,3), np.uint8)
img_space2.fill(0)
img_space3 = np.zeros((1200,500,3), np.uint8)
img_space3.fill(0)

_,contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
scale = 0.5






for i in range(2):     
    #left
    i = 16-i
    
    img_space1 = np.zeros((1200,500,3), np.uint8) 
    img_space1.fill(0)
    img_space2 = np.zeros((1200,500,3), np.uint8)
    img_space2.fill(0)
    img_space3 = np.zeros((1200,500,3), np.uint8)
    img_space3.fill(0)
    

    
    # old_height,old_width = gray_l.shape[0], gray_l.shape[1]
    # new_height,new_width = int(old_height*scale) ,int(old_width*scale)
    
    # gray_l = cv2.resize(gray_l,(new_width,new_height))
    
    _,contours, hierarchy = cv2.findContours(gray_l,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_space1,contours,i,(0,255,0),-1) #1-GT之點集
    
    area = cv2.contourArea(contours[i]) #左之面積
    
    
    #right
    
    # old_height_1,old_width_1 = gray_r.shape[0], gray_r.shape[1]
    # new_height_1,new_width_1 = int(old_height_1*scale) ,int(old_width_1*scale)
    # gray_r = cv2.resize(gray_r,(new_width_1,new_height_1))

    _,contours_1, hierarchy_1 = cv2.findContours(gray_r,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_space2,contours_1,i,(0,0,255),-1)
    
    area_1 = cv2.contourArea(contours_1[i])#右之面積
    
    #merge
    img_space3 = cv2.add(img_space1,img_space2)
    
    gray_2 = cv2.cvtColor(img_space3,cv2.COLOR_BGR2GRAY) #左右畫在空圖
    _,contours_2, hierarchy_2 = cv2.findContours(gray_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_space3,contours_2,0,(255,0,0),-1)
    
    area_2 = cv2.contourArea(contours_2[0])
    
    a_b = area + area_1 -area_2
    
    total = 2*(a_b)/(area+area_1)
    print(total)

cv2.imshow("img_space1", img_space1)
cv2.imshow("img_space2", img_space2)
cv2.imshow("img_space3",img_space3) 


cv2.waitKey(0)
cv2.destroyAllWindows()

