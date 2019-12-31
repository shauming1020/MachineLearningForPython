# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 12:22:54 2019

@author: DCMC
"""

import sys
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from FinalProject import Ui_Form   

import torch
import numpy as np
from PIL import Image

from preprocessing import clahe_hist, cv2
from resunet import ResidualUNet
from ensemble import ensemblePredict
from predict import mask_to_image, predict_img

class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.imgBt.clicked.connect(self.selectImage)
        self.ui.gtBt.clicked.connect(self.selectGT)
        self.ui.mdBt.clicked.connect(self.selectModel)
        self.ui.runBt.clicked.connect(self.run)
        
        # self.openFileNameDialog()
        self.show()
        
        self.downscale = 0.7
        self.threshold = float(self.ui.threshold.text())
        self.ensem_thre = float(self.ui.ensemThre.text())
        self.scale = float(self.ui.scale.text())
        self.dir_img = "./predict/img.png"
        self.dir_mask = "./predict/mask.png"
        self.dir_predict = "./predict/predict.png"
        self.loaded_model = "./model/BEST.pth"
        self.img = cv2.imread(self.dir_img, 0)
        self.mask = cv2.imread(self.dir_mask, 0)   
    
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()",\
                                                  "","All Files (*);;Python Files (*.py)",\
                                                      options=options)
        return fileName
    
    def selectImage(self):
        
        # self.dir_img = './predict/' + self.ui.imgName.text()
        self.dir_img = self.openFileNameDialog()
        self.img = cv2.imread(self.dir_img, 0)  
        cv2.imshow('Raw Image', cv2.resize(self.img, (int(self.downscale*500),\
                                                      int(self.downscale*1200))) )
        return 

    def selectGT(self):

        # self.dir_mask = './predict/' + self.ui.maskName.text()
        self.dir_mask = self.openFileNameDialog()
        self.mask = cv2.imread(self.dir_mask, 0)   
        cv2.imshow('Ground Truth', cv2.resize(self.mask, (int(self.downscale*500),\
                                                          int(self.downscale*1200))) )
        
        return

    def selectModel(self):
        self.ui.dcText.clear()     
        # self.loaded_model = './model/' + self.ui.modelName.text()
        self.loaded_model = self.openFileNameDialog()
        self.threshold = float(self.ui.threshold.text())
        self.scale = float(self.ui.scale.text())
        
        ## Pre Processing
        self.preimg = Image.fromarray(clahe_hist(self.img) )
        
        ## Create an UNET
        net = ResidualUNet(n_channels=1, n_classes=1)  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        
        ## Load Pre-Training Model
        net.load_state_dict(torch.load(self.loaded_model, map_location=device))
        
        self.predict = predict_img(net, self.preimg, device, self.scale,\
                                   self.threshold)
          
        ## Mask to Image
        self.predict = mask_to_image(self.predict).resize((500, 1200))
        self.predict.save(self.dir_predict)
        
        ## Find Contours
        _, contours_p, hierarchy_p = cv2.findContours(np.asarray(self.predict),\
                                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_g, hierarchy_g = cv2.findContours(np.asarray(self.mask),\
                                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours_p) == len(contours_g)) :
            self.ui.dcText.setText("Number of vertebrae match with " +\
                                   str(len(contours_g)) + " ... !\n")
            vertebrae_num = len(contours_g)
        else:
            self.ui.dcText.setText("Number of vertebrae cannot be matched ... !")
            return
        
        ## Computing dc score 
        dc_score = 0
        dc_component = "DC : \n"
        for n in range(vertebrae_num):
            i = vertebrae_num - 1 - n
            
            # compute the Area
            area_p = cv2.contourArea(contours_p[i])
            area_g = cv2.contourArea(contours_g[i])
            
            # mapping to new space
            temp1 = np.zeros((1200, 500, 3), np.uint8)
            temp1.fill(0)
            temp2 = np.zeros((1200, 500, 3), np.uint8)
            temp2.fill(0)
            cv2.drawContours(temp1, contours_p, i, (0, 255, 0), 1)
            cv2.drawContours(temp2, contours_g, i, (0, 0, 255), 1)
            
            # get the union area
            union = np.zeros((1200, 500, 3), np.uint8)
            union.fill(0)
            union = cv2.add(temp1, temp2)
            union_gray = cv2.cvtColor(union, cv2.COLOR_BGR2GRAY) 
            _, contours_u, hierarchy_u = cv2.findContours(union_gray, cv2.RETR_TREE,\
                                                        cv2.CHAIN_APPROX_SIMPLE) 
            area_u = cv2.contourArea(contours_u[0])
            
            # compute the dc score
            cross = area_g + area_p - area_u 
            total = 2 * (cross) / (area_g + area_p)
            
            dc_score += total
            dc_component += "V" + str(n) + " : " + str(round(total, 3)) + "\n"
        
        dc_avg = dc_score / vertebrae_num
        dc_component += "Average : " + str(round(dc_avg, 3))
        self.ui.dcText.setText(dc_component)
        
        ## Draw the predict contours on raw image
        self.bedraw = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(self.bedraw, contours_p, -1, (0, 0, 255), 1)
        cv2.drawContours(self.bedraw, contours_g, -1, (0, 255, 0), 1)

        ## Show    
        cv2.imshow('Result', cv2.resize(self.bedraw, (int(self.downscale*500),\
                                                      int(self.downscale*1200))) )
        
        return

    def run(self):
        self.ui.dcText.clear()
        
        ## Pre Processing
        self.preimg = Image.fromarray(clahe_hist(self.img) )
        
        ## Ensemble Predict
        self.threshold = float(self.ui.threshold.text())
        self.ensem_thre = float(self.ui.ensemThre.text())
        self.scale = float(self.ui.scale.text())
        
        self.predict = ensemblePredict(self.preimg, self.scale, self.threshold,\
                                       self.ensem_thre, N=8)
          
        ## Mask to Image
        self.predict = mask_to_image(self.predict).resize((500, 1200))
        self.predict.save(self.dir_predict)
        
        ## Find Contours
        _, contours_p, hierarchy_p = cv2.findContours(np.asarray(self.predict),\
                                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_g, hierarchy_g = cv2.findContours(np.asarray(self.mask),\
                                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours_p) == len(contours_g)) :
            self.ui.dcText.setText("Number of vertebrae match with " +\
                                   str(len(contours_g)) + " ... !\n")
            vertebrae_num = len(contours_g)
        else:
            self.ui.dcText.setText("Number of vertebrae cannot be matched ... !")
            return
        
        ## Computing dc score 
        dc_score = 0
        dc_component = "DC : \n"
        for n in range(vertebrae_num):
            i = vertebrae_num - 1 - n
            
            # compute the Area
            area_p = cv2.contourArea(contours_p[i])
            area_g = cv2.contourArea(contours_g[i])
            
            # mapping to new space
            temp1 = np.zeros((1200, 500, 3), np.uint8)
            temp1.fill(0)
            temp2 = np.zeros((1200, 500, 3), np.uint8)
            temp2.fill(0)
            cv2.drawContours(temp1, contours_p, i, (0, 255, 0), 1)
            cv2.drawContours(temp2, contours_g, i, (0, 0, 255), 1)
            
            # get the union area
            union = np.zeros((1200, 500, 3), np.uint8)
            union.fill(0)
            union = cv2.add(temp1, temp2)
            union_gray = cv2.cvtColor(union, cv2.COLOR_BGR2GRAY) 
            _, contours_u, hierarchy_u = cv2.findContours(union_gray, cv2.RETR_TREE,\
                                                        cv2.CHAIN_APPROX_SIMPLE) 
            area_u = cv2.contourArea(contours_u[0])
            
            # compute the dc score
            cross = area_g + area_p - area_u 
            total = 2 * (cross) / (area_g + area_p)
            
            dc_score += total
            dc_component += "V" + str(n) + " : " + str(round(total, 3)) + "\n"
        
        dc_avg = dc_score / vertebrae_num
        dc_component += "Average : " + str(round(dc_avg, 3))
        self.ui.dcText.setText(dc_component)
        
        ## Draw the predict contours on raw image
        self.bedraw = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(self.bedraw, contours_p, -1, (0, 0, 255), 1)
        cv2.drawContours(self.bedraw, contours_g, -1, (0, 255, 0), 1)

        ## Show    
        cv2.imshow('Result', cv2.resize(self.bedraw, (int(self.downscale*500),\
                                                      int(self.downscale*1200))) )
        
        return
    
    
app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())


