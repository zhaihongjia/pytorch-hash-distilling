'''

This file contains the margin based loss funtions:

Ang_Margin: ||x||cos(m*theta)
Add_Margin: cos(theta)-m
Arc_Margin: cos(theta+m)
Norm: cos(theta)
Hash_Margin: ||x||(cos(theta)-m)

'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

import math
import errno
import os

#---------- Margin based functions: the output contains the margin
#----------||x||cos(m*theta)
def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class Ang_Margin(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.it=0
        self.LambdaMin=5.0
        self.LambdaMax=1500.0
        self.lamb=1500.0
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input, label):
        label = torch.zeros(label.size(0), 10).scatter_(1, label.cpu().unsqueeze(1), 1)
        label=label.cuda()
        x = input   # size=(B,F)    F is feature length   bachsize x feature length
        w = self.weight # size=(F,Classnum)

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=Bachsize  norm
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum  norm

        cos_theta = x.mm(ww) # size=(B,Classnum)  b,f x f,classnum-> b, classnum
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)

        #----- compute the changed output logits
        self.it+=1
        index=(label==1).byte()
        index=Variable(index)
        self.lamb=max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it))
        output=cos_theta*1.0
        output[index]-=cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        return output

# --------- cos(theta)-m
class Add_Margin(nn.Module):
    def __init__(self, in_features, out_features,s=30.0, m=0.40):
        super(Add_Margin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        label = torch.zeros(label.size(0), 10).scatter_(1, label.cpu().unsqueeze(1), 1)
        label=label.cuda()
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine #- self.m
        output = (label.float() * phi) + ((1.0 - label.float()) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


# --------- cos(theta+m)
class Arc_Margin(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(Arc_Margin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        label = torch.zeros(label.size(0), 10).scatter_(1, label.cpu().unsqueeze(1), 1)
        label=label.cuda()
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))   # mol-- by default bias=None
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (label.float() * phi) + ((1.0 - label.float()) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


#---------- no margin
class Norm(nn.Module):
    def __init__(self, in_features, out_features):
        super(Norm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        label = torch.zeros(label.size(0), 10).scatter_(1, label.cpu().unsqueeze(1), 1)
        label=label.cuda()
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine


# ---------||x||(cos(theta)-m)
class Hash_Margin(nn.Module):
    def __init__(self, in_features, out_features, m=0.40):
        super(Hash_Margin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        label = torch.zeros(label.size(0), 10).scatter_(1, label.cpu().unsqueeze(1), 1)
        label=label.cuda()
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        feature_norm=input.pow(2).sum(1).pow(0.5)
        output = (label.float() * phi) + ((1.0 - label.float()) * cosine)
        output *= feature_norm.unsqueeze(1).expand_as(output)

        return output


# feature regularizations
def guassian(x,weight,theta):
    loss=weight/(theta*np.sqrt(2*np.pi))*(1-torch.exp((torch.pow(x,2))/(2*theta*theta)))
    return loss

def randlsr(label,phi):
    soft=label.cpu()
    total=soft.sum(dim=1)
    p=total.unsqueeze(1).expand_as(soft)
    noise_label=torch.rand((label.size(0),10))*phi/10
    func=torch.nn.Softmax(dim=1)
    soft=func(soft*(1-phi)/p+noise_label)
    soft=soft*p
    return soft.cuda()

#---------- Compute the loss with/without label smoothing for the margin based logits
class Multi_Loss(nn.Module):
    def __init__(self,smooth=0,scale=3,lam=0.5,eps=0.0,regularization=0):
        super(Multi_Loss,self).__init__()
        self.regularization=regularization
        self.scale=scale
        self.lam=lam
        self.smooth=smooth
        self.eps=eps

    def forward(self,features,output,target):
        target = torch.zeros(target.size(0), 10).scatter_(1, target.cpu().unsqueeze(1), 1)
        target=target.cuda()
        if self.smooth==0:
            # compute the loss
            logpt = F.log_softmax(output,dim=1)
            logpt=(logpt*(target.float())).sum(dim=1) # b x classnum ---> b x 1
            logpt = logpt.view(-1)
            loss=-1*logpt

        elif self.smooth==1:
            soft=target.cpu()
            total=soft.sum(dim=1)
            soft=(self.scale*soft.float())/total.float().unsqueeze(1)
            soft=soft.cuda()

            # compute the loss
            logpt = F.log_softmax(output,dim=1)
            logpt=(logpt*soft).sum(dim=1) # b x classnum ---> b x 1
            logpt = logpt.view(-1)
            loss=-1*logpt

        elif self.smooth==2:
            soft=target.cpu()
            total=soft.sum(dim=1)
            noise_label=(torch.ones((target.size(0),10))-target.cpu().float())/10
            noise_label=noise_label*(total.unsqueeze(1).expand_as(noise_label))
            soft=(((1-self.eps)*target.cpu().float())/total.unsqueeze(1).expand_as(noise_label)+self.eps*noise_label)*total.unsqueeze(1).expand_as(noise_label)
            soft=soft.cuda()

            # compute the loss
            logpt = F.log_softmax(output,dim=1)
            logpt=(logpt*soft).sum(dim=1) # b x classnum ---> b x 1
            logpt = logpt.view(-1)
            loss=-1*logpt

        elif self.smooth==3:
            soft=randlsr(target,self.eps)
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
