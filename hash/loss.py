import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


#---------- Margin based functions: the output contains the margin
#----------||x||cos(m*theta)
class AA_Margin(nn.Module):
    def __init__(self, in_features, out_features, m = 1):
        super(AA_Margin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature length   bachsize x feature length
        w = self.weight # size=(F,Classnum)

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=Bachsize  norm
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum  norm

        cos_theta = x.mm(ww) # size=(B,Classnum)  b,f x f,classnum-> b, classnum
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        cos_m_theta = self.mlambda[self.m](cos_theta)
        output=torch.min(cos_theta,cos_m_theta)
        return output


#---------- Compute the loss with/without label smoothing for the margin based logits
class Multi_Loss(nn.Module):
    def __init__(self,class_num=21,smooth=0,eps=0.0,regularization=0,lam=0.5):
        super(Multi_Loss,self).__init__()
        self.regularization=regularization
        self.lam=lam
        self.smooth=smooth
        self.eps=eps
        self.class_num=class_num

    def forward(self,features,output,target):
        if self.smooth==0:
            # compute the loss
            logpt = F.log_softmax(output,dim=1)
            logpt=(logpt*(target.float())).sum(dim=1) # b x classnum ---> b x 1
            logpt = logpt.view(-1)
            loss=-1*logpt

        elif self.smooth==1:
            soft=target.cpu()
            total=soft.sum(dim=1)
            noise_label=(torch.ones((target.size(0),self.class_num))-target.cpu().float())/self.class_num
            noise_label=noise_label*(total.unsqueeze(1).expand_as(noise_label))
            #soft=(((1-self.eps)*target.cpu().float())/total.unsqueeze(1).expand_as(noise_label)+self.eps*noise_label)*total.unsqueeze(1).expand_as(noise_label)
            soft=((1-self.eps)*target.cpu().float())+self.eps*noise_label
            soft=soft.cuda()

            # compute the loss
            logpt = F.log_softmax(output,dim=1)
            logpt=(logpt*soft).sum(dim=1) # b x classnum ---> b x 1
            logpt = logpt.view(-1)
            loss=-1*logpt

        if self.regularization==1:
            loss+=self.lam*((features.abs()-1).abs().mean()) # l1  regularization
        elif self.regularization==2:
            loss+=self.lam*(torch.pow(features.abs()-1,2).mean()) # l2  regularization

        return loss.mean()
