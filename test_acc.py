import torch
import torchvision
from torchvision import models,transforms,datasets

import torch.nn as nn
from torch.autograd import Variable

from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent
from utils import trainloader,testloader

#---------------load test model------------------
bits=48
modelpath="./models/teacher/"
#-------teacher model----------------------------
net=Resnet18PlusLatent(bits)
net.load_state_dict(torch.load(modelpath))
net.eval()
#------------------------------------------------

correct=0
total=0
for inputs,labels in trainloader:
    inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
    _,_,_,result=net(inputs)
    predict=torch.max(result.data,1)[1]
    total+=labels.size()[0]
    correct+=(predict==labels).sum()
print("train: total:{}  correct:{}".format(total,correct))

correct=0
total=0
for t_in,t_labels in testloader:
    t_in,t_labels=Variable(t_in.cuda()),Variable(t_labels.cuda())
    _,_,_,t_result=net(t_in)
    t_prediction=torch.max(t_result.data,1)[1]
    total+=t_labels.size()[0]
    correct+=(predict==labels).sum()
print("test: total:{}  correct:{}".format(total,correct))
