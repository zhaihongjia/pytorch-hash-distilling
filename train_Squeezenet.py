import torch
import torchvision
from torchvision import models,transforms,datasets
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent
from utils import trainloader,testloader


#----------high parameters
MOMENTUM=0.9
LR=0.01
EPOCH=50000
#----------models   optimizer   loss_function
net=torchvision.models.squeezenet1_1(pretrained=True)
net.classifier = nn.Sequential(net.classifier[0], nn.Conv2d(512, 10, kernel_size = (1,1), stride = (1,1)), net.classifier[2], nn.AvgPool2d(kernel_size = 1, stride =1))
net.num_classes = 10
net.cuda()

optimizer_sgd=optim.SGD(net.parameters(),lr=LR, momentum=MOMENTUM, weight_decay=0.0005)
#optimizer_adam=optim.Adam(student.parameters(),lr=LR, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[40], gamma=0.1)

loss_ce=nn.CrossEntropyLoss().cuda()
#loss_mse=nn.MSELoss().cuda()

#----------------train and test------------------
best=0
for i in torch.arange(1,EPOCH+1):
    scheduler.step()
    net.train()
    train_loss=0.0
    correct=0
    total=0
    for inputs,labels in trainloader:
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
        result=net(inputs)
        loss=loss_ce(result,labels)
        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()
        train_loss+=loss.data
        predict=torch.max(result.data,1)[1]
        total+=labels.size()[0]
        correct+=(predict==labels).cpu().sum()
        #print("loss:",loss)
    print("epoch:{}  loss:{}  total:{}  correct:{}".format(i,train_loss,total,correct))

    #---------------test-----------
    total=0
    correct=0
    for t_inputs,t_labels in testloader:
        t_inputs,t_labels=Variable(t_inputs.cuda()),Variable(t_labels.cuda())
        t_result=net(t_inputs)
        _,t_predict=torch.max(t_result.data,1)
        total+=t_labels.size(0)
        correct+=(t_predict==t_labels).cpu().sum()
    print("test:  total:{}  correct:{}".format(total,correct))

    if i%5==0:
        print("Saving model-------------------------!")
        torch.save(net.state_dict(),"./models/Squeezenet/epoch{}_{}.pkl".format(i,correct))
