'''
stage three: distilling student model according to two fc layers
'''
import torch
import torchvision
from torchvision import models,transforms,datasets
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent
from utils import trainloader,testloader

#-------------change bits to:12 24 36 48---------------------------
bits=12
#----------------------------------------------------------------

EPOCH=8000
LR=0.01

#------------------load models---------------------------------
student=SqueezenetPlusLatent(bits)
#student.load_state_dict(torch.load("./models/student/adam_mse_epoch8000.0_222.pkl"))
student.cuda()
student.train(True)
teacher=Resnet18PlusLatent(bits)
teacher.load_state_dict(torch.load("./models/teacher/T_bit{}.pkl".format(bits)))
teacher.cuda()
teacher.train(False)
#---------------------------------------------------------------

#loss_mse=nn.MSELoss(size_average=False).cuda()
loss_mse=nn.MSELoss().cuda()
#loss_ce=nn.CrossEntropyLoss().cuda()
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
        l=l+labels.size()[0]
        outputs1,_,outputs2,_=student(inputs)
        t_out1,_,t_out2,_=teacher(inputs)
        t_out22=t_out2.detach()
        t_out11=t_out1.detach()
        #-------------------------------------------------------
        #t_out22=torch.round(torch.clamp(t_out22,min=0))
        #t_out11=torch.round(torch.clamp(t_out11,min=0))
        #t_out11=t_out11.type(torch.cuda.LongTensor)[1]
        #t_out22=t_out22.type(torch.cuda.LongTensor)[1]
        #loss=loss_ce(result,labels)

        #--------MSELoss---without distilling---
        #print('outp',outputs2)
        #print('tout',t_out22)
        #print('loss.data',loss.data)
        loss=loss_mse(outputs2,t_out22)+loss_mse(outputs1,t_out11)
        #loss=loss_ce(outputs2,t_out22)+loss_ce(outputs1,t_out11)

        optimer.zero_grad()
        loss.backward()
        optimer.step()
        train_loss+=loss.data
    print("epoch:{}  loss:{}  total:{}  correct:{}".format(i,train_loss))

    if i%20==0:
        print("Saving model-------------------------!")
        torch.save(student.state_dict(),"./models/student/3{}/S_bit{}_epoch{}.pkl".format(bits,bits,i))
