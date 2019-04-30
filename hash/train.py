import sys
import torchvision
import torch
import os
from torch.autograd import Variable
from torch import optim
from torch.utils import data
import argparse,datetime,time
import numpy as np

from dataset import load_data
from loss import Multi_Loss
from utils import *
from net.Network import *

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
    defile.write("bit:{}  epoch:{}  loss:{}  lr:{}\n".format(len,i,train_loss,LR))
    defile.flush()


    if i%args.save_iter==0: #and i>=100:
        mkdir_if_missing(args.savepath+'{}/'.format(bits))
        filename=args.savepath+'{}/'.format(bits)+'epoch{}.pkl'.format(i)
        state = model.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(state, filename)
        print("----------->save model in {}\n".format(filename))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='DHH')
    parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--weight_decay',default=5e-5,type=float)
    parser.add_argument('--save_iter',default=10,type=int)
    parser.add_argument('--seed',default=1,type=int)
    parser.add_argument('--change_mode',default=1,type=int)
    parser.add_argument('--batch',default=128,type=int)
    # debug information
    parser.add_argument('--epochs',default=450,type=int)
    parser.add_argument('--savepath',default='./models/',help='the saving path for trained models')
    parser.add_argument('--logfile',default='debug.txt')
    # label smoothing regularization
    parser.add_argument('--eps',default=0.0,type=float)
    parser.add_argument('--smooth',default=0,type=int)
    # adaptive angular margin
    parser.add_argument('--m',default=1,type=int)
    # feature regularization
    parser.add_argument('--lam',default=0.5,type=float)
    parser.add_argument('--regularization',default=0,type=int)
    # according to the dataset
    parser.add_argument('--class_num',default=21,type=int)
    parser.add_argument('--dataset',default=nus,type=str)
    args=parser.parse_args()

    torch.cuda.set_device(0)
    torch.backends.cudnn.bencmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # dataset
    trainloader=load_data(args.dataset,'jpg',args.batchsize)

    # choose the change epoches
    if args.change_mode==1:
        step_epoches={80,190,290,380}  #120,240,300  --->cifar10
    elif args.change_mode==2:
        step_epoches={180,280}

    # debug file ==> add some head information in the file
    defile=open('./logfile/'+args.logfile,"a",newline="\n")

    # configuration information
    print('========>:Config Start!')
    print("Label smoothing regularization ===>","Use LSR, eps:"+str(args.eps)+"!" if args.smooth!=0 else "No LSR!")
    print("Adative angular margin: ===>","Use AA_Margin, m:"+str(args.m)+"!" if args.m!=1 else "No AA_Margin!")
    print("Feature regularization: ===>","Use Feature regularization: type:{} lam:{}!".format(args.regularization,args.lam) if args.regularization!=0 else "No Feature regularization!")
    print('========>:Config End!')

    use_cuda=torch.cuda.is_available()
    for bits in {16,32,48,64}: # 16,32,48,64
        model=AlexNet(bits,args.class_num,args.m)
        #------------ the loss mode and label smooth:  class_num=21,smooth=0,eps=0.0,regularization=0,lam=0.5
        criterion=Multi_Loss(class_num=args.class_num,smooth=args.smooth,eps=args.eps,regularization=args.regularization,lam=args.lam)
        if use_cuda:
            model.cuda()
            criterion.cuda()
        LR=args.lr
        optimizer=torch.torch.optim.SGD(model.parameters(),lr=LR,momentum=args.momentum, weight_decay=args.weight_decay)

        for i in range(1,args.epochs+1):
            if i in step_epoches:
                LR=LR*0.1
                #optimizer=torch.torch.optim.SGD(model.parameters(),lr=LR,momentum=args.momentum, weight_decay=args.weight_decay)
                for param_groups in optimizer.param_groups:
                    param_groups['lr']=LR
            train(i)
    defile.close()
