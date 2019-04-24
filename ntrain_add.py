import sys
from nmodel import DSH
import torchvision
import torch
import os
from torch.autograd import Variable
from torch import optim
torch.backends.cudnn.bencmark = True
import argparse,datetime,time
torch.cuda.set_device(0)
import numpy as np

from nNUS_AUG import trainloader,testloader
from nmloss import *

def train(i):
    model.train()
    train_loss=0.0
    for inputs,targets in trainloader:
        optimizer.zero_grad()
        if use_cuda:
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        else:
            inputs, targets = Variable(inputs), Variable(targets)
        features,outputs = model(inputs, targets)
        loss = criterion(features,outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data
    print("bit:{}  epoch:{}  loss:{}  lr:{}\n".format(len,i,train_loss,LR))
    

    if i%args.save_iter==0 and i>=100:
        mkdir_if_missing(args.savepath+'{}/'.format(len))
        filename=args.savepath+'{}/'.format(len)+'epoch{}.pkl'.format(i)
        state = model.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(state, filename)
        print("----------->save model in {}\n".format(filename))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train DSH model on NUS-WIDE dataset using Addtive Margin')
    parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--weight_decay',default=5e-5,type=float)
    parser.add_argument('--save_iter',default=10,type=int)
    parser.add_argument('--seed',default=1,type=int)
    # something will be changed
    parser.add_argument('--logfile',default='add_margin_debug.txt')  # echo the debug info to the txt file
    parser.add_argument('--eps',default=0.1,type=float)
    parser.add_argument('--smooth',default=0,type=int) # the switch to control label smooth
    parser.add_argument('--savepath',default='./Add_margin/',help='the saving path for trained models')  # the path to save the model
    parser.add_argument('--epochs',default=450,type=int)# the total training epoches
    parser.add_argument('--change_mode',default=1,type=int)  # the learning rate change mode info according to the epoch
    parser.add_argument('--info',default='the running infomation',type=str) # echo the information of the program
    parser.add_argument('--loss_mode',default=1,type=int) # different Margin mode
    parser.add_argument('--lam',default=0.5,type=float)
    parser.add_argument('--regularization',default=0,type=int)
    parser.add_argument('--scale',default=3.0,type=float)
    args=parser.parse_args()

    # choose the change epoches
    if args.change_mode==1:
        step_epoches={80,190,290,380}  #120,240,300  --->cifar10
    elif args.change_mode==2:
        step_epoches={180,280}

    # fix the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # debug file  add some head information in the file
    defile=open('./logfile/'+args.logfile,"a",newline="\n")
    defile.write(args.info+'\n')
    defile.flush()

    # to check smooth mode
    if args.smooth==True:
        print('Using Label smooth!')
    else:
        print('Not using Label smooth')

    use_cuda=torch.cuda.is_available()
    for len in {12,24,36,48}:
        model=DSH(len,args.loss_mode)
        #------------ the loss mode and label smooth  smooth=0,scale=3,lam=0.5,eps=0.0,regularization=0):
        criterion=Multi_Loss(smooth=args.smooth,scale=args.scale,lam=args.lam,eps=args.eps,regularization=args.regularization)
        if use_cuda:
            model.cuda()
            criterion.cuda()
        LR=args.lr
        optimizer=torch.torch.optim.SGD(model.parameters(),lr=LR,momentum=args.momentum, weight_decay=args.weight_decay)

        # start training time
        strattime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
        for i in range(1,args.epochs+1):
            if i in step_epoches:
                LR=LR*0.1
                #optimizer=torch.torch.optim.SGD(model.parameters(),lr=LR,momentum=args.momentum, weight_decay=args.weight_decay)
                for param_groups in optimizer.param_groups:
                    param_groups['lr']=LR
            train(i)
        #end training time
        endtime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
        # echo the infomation
        print('Start training time: {} \nEnd training time: {}'.format(strattime,endtime))
    defile.close()
