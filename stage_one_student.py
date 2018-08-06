import torch
import torchvision
from torchvision import models,transforms,datasets
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent
from utils import trainloader,testloader

EPOCH=50000
LR=0.01
#-----------change bits to:12 24 36 48---------------------------
bits=12
#----------------------------------------------------------------
#------------------load models---------------------------------
student=SqueezenetPlusLatent(bits)
#student.load_state_dict(torch.load("./models/fixstudent/acc_epoch3000.0_8368.pkl"))
student.cuda()
student.train(True)

loss_ce=nn.CrossEntropyLoss().cuda()
#loss_l1l=nn.L1Loss().cuda()
#optimer=optim.Adam(student.parameters(),lr=LR)
optimer=optim.SGD(student.parameters(),lr=LR,momentum=0.9)
scheduler=optim.lr_scheduler.StepLR(optimer,step_size=40,gamma=0.1)

#-----------------for train and test-----------------------------
for i in torch.arange(0,EPOCH+1):
    scheduler.step()

    train_loss=0.0
    correct=0
    total=0
    for inputs,labels in trainloader:
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
        outputs1,_,outputs2,result=student(inputs)
        loss=loss_ce(result,labels)

        optimer.zero_grad()
        loss.backward()
        optimer.step()
        train_loss+=loss.data
        predict=torch.max(result.data,1)[1]
        total+=labels.size()[0]
        correct+=(predict==labels).sum()
    print("epoch:{}  loss:{}  total:{}  correct:{}".format(i,train_loss,total,correct))

    #---------------test-----------
    total=0
    correct=0
    for t_inputs,t_labels in testloader:
        t_inputs,t_labels=Variable(t_inputs.cuda()),Variable(t_labels.cuda())
        _,_,_,test_result=student(t_inputs)
        _,t_predict=torch.max(test_result.data,1)
        total+=t_labels.size(0)
        correct+=(t_predict==t_labels).sum()
    print("test:  total:{}  correct:{}".format(total,correct))

    if i%10==0:
        print("Saving model-------------------------!")
        torch.save(student.state_dict(),"./models/student/1{}/acc_epoch{}_{}.pkl".format(bits,i,correct))
