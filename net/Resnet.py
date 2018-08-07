import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

resnet18=models.resnet18()
resnet18.fc = nn.Linear(in_features = 512, out_features = 10)
resnet18.load_state_dict(torch.load("./models/resnet18/pre_resnet18.pkl"))

class Resnet18PlusLatent(nn.Module):
    def __init__(self,bits):
        super(Resnet18PlusLatent,self).__init__()
        self.conv1=resnet18.conv1
        self.bn1=resnet18.bn1
        self.relu=resnet18.relu
        self.maxpool=resnet18.maxpool
        self.layer1=nn.Sequential(*list(resnet18.layer1.children()))
        self.layer2=nn.Sequential(*list(resnet18.layer2.children()))
        self.layer3=nn.Sequential(*list(resnet18.layer3.children()))
        self.layer4=nn.Sequential(*list(resnet18.layer4.children()))
        self.avgpool=nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.fc=nn.Linear(in_features=512, out_features=256, bias=True)
        self.bits=bits
        self.Linear1=nn.Linear(256, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2=nn.Linear(self.bits, 256)
        self.Linear3=nn.Linear(256,10)
    def forward(self,x):
        x=self.layer4(self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x))))))))
        x=self.avgpool(x)
        x=x.view(x.size(0),512)
        x=self.sigmoid(((x-x.mean(1).unsqueeze(1).expand(x.size()[0],x.size()[1]))/torch.sqrt(x.var(1).unsqueeze(1).expand(x.size()[0],x.size()[1]))))
        former=self.fc(x)
        former=self.sigmoid(((former-former.mean(1).unsqueeze(1).expand(former.size()[0],former.size()[1]))/torch.sqrt(former.var(1).unsqueeze(1).expand(former.size()[0],former.size()[1]))))
        features=self.Linear1(former)
        features=self.sigmoid(((features-features.mean(1).unsqueeze(1).expand(features.size()[0],features.size()[1]))/torch.sqrt(features.var(1).unsqueeze(1).expand(features.size()[0],features.size()[1]))))
        latter=self.Linear2(features)
        latter=self.sigmoid(((latter-latter.mean(1).unsqueeze(1).expand(latter.size()[0],latter.size()[1]))/torch.sqrt(latter.var(1).unsqueeze(1).expand(latter.size()[0],latter.size()[1]))))
        result=F.softmax(self.Linear3(latter),dim=1)
        return former,features,latter,result


# class Resnet50PlusLatent(nn.Module):
#     def __init__(self,bits):
#         super(Resnet50PlusLatent,self).__init__()
#         self.bits=bits
#         self.Linear1=nn.Linear(10, self.bits)
#         self.sigmoid = nn.Sigmoid()
#         self.Linear2=nn.Linear(self.bits, 10)
#     def forward(self,x):
#         former=resnet50(x)
#         features=self.sigmoid(self.Linear1(former))
#         result=self.Linear2(features)
#         return former,features,result
