# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 23:29:10 2022

@author: ZRC
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


    
class REDCNNNet(nn.Module):
    def __init__(self):
        super(REDCNNNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=3,stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=3,stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3,stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=3,stride=1, padding=1),
            nn.ReLU()
        )
        self.invconv5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.invconv6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.invconv7 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.invconv8 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
             nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
    def forward(self, x):
#        encode
        resi_dual_1 = x
        out = self.conv1(x)
        resi_dual_2 = out
        out = self.conv2(out)
        resi_dual_3 = out
        out = self.conv3(out)
        resi_dual_4 = out
        out = self.conv4(out)   
        out = self.invconv5(out)
        out = out+resi_dual_4
        
#        decode
        out = self.invconv6(out)
        out = out+resi_dual_3
        out = self.invconv7(out)
        out = out+resi_dual_2
        out = self.invconv8(out)
        out = out+resi_dual_1
        out = self.conv9(out)
        
        return out
        