import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
from collections import OrderedDict
from loss import AA_Margin

class AlexNet(nn.Module):
  def __init__(self, bits, class_num, m=1):
    super(AlexNet, self).__init__()
    self.bits=bits
    self.class_num=class_num
    self.m=m
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    #self.feature_layers = nn.Sequential(self.features, self.classifier)
    self.hash_layer = nn.Linear(model_alexnet.classifier[6].in_features, self.bits)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.relu=nn.Sigmoid()
    self.margin=AA_Margin(self.bits,self.class_num,self.m)

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    features = self.hash_layer(x)
    out=self.margin(self.relu(features))
    #print(features)
    return features,out



class DSH(nn.Module):
    def __init__(self,bits,class_num,m=1):
        super(DSH,self).__init__()
        self.bits=bits
        self.class_num=class_num
        self.m=m

        self.features=nn.Sequential(OrderedDict([
        # first conv layer
        ('conv1',nn.Conv2d(3,32,kernel_size=5,stride=1,padding=0,bias=True)),
        ('bn1',nn.BatchNorm2d(32)),
        ('pool1',nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)),
        ('relu1',nn.ReLU(inplace=True)),

        # second conv layer
        ('conv2',nn.Conv2d(32,32,kernel_size=5,stride=1,padding=0,bias=True)),
        ('bn2',nn.BatchNorm2d(32)),
        ('pool2',nn.AvgPool2d(kernel_size=3,stride=2,ceil_mode=True)),
        ('relu2',nn.ReLU(inplace=True)),

        # third conv layer
        ('conv3',nn.Conv2d(32,64,kernel_size=5,stride=1,padding=0,bias=True)),
        ('bn3',nn.BatchNorm2d(64)),
        ('pool3',nn.AvgPool2d(kernel_size=3,stride=2,ceil_mode=True)),
        ('relu3',nn.ReLU(inplace=True)),
        ]))

        self.linear=nn.Sequential(OrderedDict([
        ('linear1',nn.Linear(1024,500)),
        ('relu',nn.ReLU(inplace=True)),
        ('linear',nn.Linear(500,self.bits)),
        ('drop',nn.Dropout(0.1)),
        ]))

        self.margin=AA_Margin(self.bits,self.class_num,self.m)

    def forward(self,x):
        features=self.features(x)
        features=self.linear(features)
        out=self.margin(features)
        return features,out























#  end file
