import torch
import torchvision
from torchvision import models,transforms,datasets
import torch.nn as nn
from torch.autograd import Variable
from net.Resnet import Resnet18PlusLatent
from utils import trainloader,testloader

#-----------change bits to:12 24 36 48---------------------------
bits=36
#----------------------------------------------------------------
MOMENTUM=0.9
LR=0.01
EPOCH=80000

#------------------load data-----------------------
model=Resnet18PlusLatent(bits)
#model.load_state_dict(torch.load('./models/teacher/T_bit12_epoch31.0_9465.pkl'))
model.cuda()
loss_function=nn.CrossEntropyLoss().cuda()
optimer=torch.optim.SGD(model.parameters(),lr=LR, momentum=MOMENTUM, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimer, milestones=[40], gamma=0.1)

#------------------train --------------------------
best=0
for i in torch.arange(50,EPOCH+1):
    model.train()
    train_loss=0.0
    total=0
    correct=0
    for inputs,labels in trainloader:
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
        _,_,_,outputs=model(inputs)
        loss=loss_function(outputs,labels)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        train_loss+=loss.data
        predict=torch.max(outputs.data,1)[1]
        total+=labels.size()[0]
        correct+=(predict==labels).sum()
        #print("loss:",loss)
    print("epoch:{}  loss:{}  total:{}  correct:{}".format(i,train_loss,total,correct))

    total=0
    correct=0
    for t_inputs,t_labels in testloader:
        model.eval()
        t_inputs,t_labels=Variable(t_inputs.cuda()),Variable(t_labels.cuda())
        _,_,_,t_outputs=model(t_inputs)
        t_predict = torch.max(t_outputs.data, 1)[1]
        total+=t_labels.size(0)
        correct+=(t_predict==t_labels).sum()
    print("Test: total:{}  correct:{}".format(total,correct))
    if correct>best or i%10==0:
        best=correct
        print("Saving model-------------------------!")
        torch.save(model.state_dict(),"./models/teacher/T_bit{}_epoch{}_{}.pkl".format(bits,i,correct))
