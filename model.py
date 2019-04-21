'''
This file defines the model we used in our experiments for NUS-WIDE dataset
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from mloss import *

class DSH(nn.Module):
    def __init__(self,bits,mode=1):
        super(DSH,self).__init__()
        self.bits=bits
        self.mode=mode

        # the input shape if linear1 is calculated manually
        # shape after conv layer or pooling layer  = (input_width+2*pad-pool_size)/stride+1
        self.features = nn.Sequential(OrderedDict([
            # first conv layer
            ('conv1', nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=True)),
            ('batchnorm1', nn.BatchNorm2d(32)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ('relu1', nn.ReLU(inplace=True)),

            # second conv layer
            ('conv2', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=True)),
            ('batchnorm2', nn.BatchNorm2d(32)),
            ('pool2', nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ('relu2', nn.ReLU(inplace=True)),

            # third conv layer
            ('conv3', nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True)),
            ('batchnorm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True))
        ]))

        # two fully connected layer
        self.linear = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1024, 500)), #1024 for 32x32 4096
            ('relu', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(500, self.bits)),
            ('drop', nn.Dropout(0.1)),
        ]))

        # mode=1  softmax
        # mode=2  Ang_Margin
        # mode=3  Add_margin
        # mode=4  Arc_margin
        # mode=5  Norm
        # mode=6  Hash_margin
        if self.mode==1:
            self.classification_layer=nn.Linear(self.bits,10)
        elif self.mode==2:
            self.classification_layer=Ang_Margin(self.bits,10)
        elif self.mode==3:
            self.classification_layer=Add_Margin(self.bits,10)
        elif self.mode==4:
            self.classification_layer=Arc_Margin(self.bits,10)
        elif self.mode==5:
            self.classification_layer=Norm(self.bits,10)
        else:
            self.classification_layer=Hash_Margin(self.bits,10)

    def forward(self, x,label):
        features = self.features(x)
        features = self.linear(features.view(x.size(0), -1))
        if self.mode==1:
            out=self.classification_layer(features)
        else:
            out=self.classification_layer(features,label)
        return features,out
