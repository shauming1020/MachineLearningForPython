""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .resunet_parts import *


class ResidualUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResidualUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResidualDoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 128, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.up5 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        # self.inc = ResidualDoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 1024)
        # self.down4 = Down(1024, 1024)
        # self.up1 = Up(2048, 1024, bilinear)
        # self.up2 = Up(1280, 256, bilinear)
        # self.up3 = Up(384, 128, bilinear)
        # self.up4 = Up(192, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        
        # self.inc = ResidualDoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 512)
        # self.down3 = Down(512, 512)
        # self.down4 = Down(512, 512)
        # self.down5 = Down(512, 1024)
        # self.up1 = Up(1536, 512, bilinear)
        # self.up2 = Up(1024, 512, bilinear)
        # self.up3 = Up(1024, 128, bilinear)
        # self.up4 = Up(256, 64, bilinear)
        # self.up5 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        
        return logits

        
