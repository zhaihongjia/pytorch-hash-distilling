import torch
import torchvision

class ResNetHash(torch.nn.Module):
    def __init__(self,name,bits,classnum):
        super(ResNetHash,self).__init__()
        self.name=name
        self.bits=bits
        self.classnum=classnum
        if self.name=="resnet18":
            self.model=torchvision.model.resnet18(True)
            print("======>> Load resnet18!")
        elif self.name=="resnet34":
            self.model=torchvision.model.resnet34(True)
            print("======>> Load resnet34!")
        elif self.name=="resnet50":
            self.model=torchvision.model.resnet50(True)
            print("======>> Load resnet50!")
        elif self.name=="resnet101":
            self.model=torchvision.model.resnet101(True)
            print("======>> Load resnet101!")
        elif self.name=="resnet152":
            self.model=torchvision.model.resnet152(True)
            print("======>> Load resnet152!")
        self.model.fc=torch.nn.Linear(512,self.bits)
        self.remain=nn.Sequential(*list(self.model.children()))
        self.cla=torch.nn.Linear(self.bits,self.classnum)
        self.sigmoid =torch.nn.Sigmoid()


    def forward(self,x):
        x=self.remain(x)
        x=x.view(x.size()[0],512)
        features=self.fc(x)
        x=self.cla(self.sigmoid(features))
        return features,x





















#
