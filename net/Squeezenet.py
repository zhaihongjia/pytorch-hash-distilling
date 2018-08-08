#----------squeezenetPlusLatent----------------
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

squeezenet=models.squeezenet1_1(pretrained=True)
#squeezenet=models.squeezenet1_1()
squeezenet.classifier[1]=nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
squeezenet.classifier[3]=nn.AvgPool2d(kernel_size=1, stride=1, padding=0)

class SqueezenetPlusLatent(nn.Module):
    def __init__(self,bits):
        super(SqueezenetPlusLatent,self).__init__()
        self.remain=nn.Sequential(*list(squeezenet.children()))
        self.bits=bits
        self.Linear1=nn.Linear(256, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2=nn.Linear(self.bits, 256)
        self.Linear3=nn.Linear(256,10)


    def forward(self,x):
        former=self.remain(x)
        former=former.view(former.size(0),256)
        features=self.Linear1(F.dropout(F.relu(former),0.5))
        latter=self.Linear2(F.dropout(F.relu(features),0.5))
        result=self.Linear3(latter)
        return former,features,latter,result
