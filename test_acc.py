import os
import torch
import torchvision
from torchvision import models,transforms,datasets

import torch.nn as nn
from torch.autograd import Variable

from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent
from utils import trainloader,testloader

#---------------load test model------------------
target_root="./models/teacher/"
models=os.listdir(target_root)
modelspath=[os.path.join(target_root,model) for model in models if model.endswith("pkl")]
for modelpath in modelspath:
    #------judge bits-----------
    if "bit12" in modelpath:
        bits=12
    elif "bit24" in modelpath:
        bits=24
    elif "bit36" in modelpath:
        bits=36
    elif "bit48" in modelpath:
        bits=48

    #-------teacher model------only teacher need test acc---------
    net=Resnet18PlusLatent(bits)
    net.load_state_dict(torch.load(modelpath))
    net.cuda()
    net.eval()
    #------------------------------------------------
    print("model:",modelpath)
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
        t_predict=torch.max(t_result.data,1)[1]
        total+=t_labels.size()[0]
        correct+=(t_predict==t_labels).sum()
    print("test: total:{}  correct:{}".format(total,correct))
