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
import lr_schedule

def train():
    # training data length
    len_train=len(load_data('train',args.dataset,args.batchsize))-1

    # load network
    model=AlexNet(args.bits,args.class_num,args.m)

    # loss function
    criterion=Multi_Loss(class_num=args.class_num,smooth=args.smooth,\
    eps=args.eps,regularization=args.regularization,lam=args.lam)

    # collect parameters
    parameter_list = [{"params":model.features.parameters(), "lr":1}, \
                      {"params":model.hash_layer.parameters(), "lr":10} \
                      {"params":model.margin.parameters(), "lr":10}]

    # set optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9, \
    weight_decay=0.0005,nesterov=True)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])


    # train
    for i in range(iterations):
        # save parameters in iter
        if (i+1)%2500==0:
            path=args.savepath+'{}/'.format(args.bits)
            if not os.path.exists(path):
                os.mkdir(path)
            torch.save(model.cpu.state_dict(),path+"iter_{}".format(i+1))

        # train one iter
        model.train(True)
        schedule_param={"init_lr":args.lr, "gamma":0.5, "step":2000}
        optimizer = lr_schedule.step_lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i%len_train==0:
            trainloader=iter(load_data('train',args.dataset,args.batchsize))
        inputs,targets=trainloader.next()
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        features,outputs = model(inputs)
        loss = criterion(features,outputs,targets)
        loss.backward()
        optimizer.step()

        print("Iter:{:05d}  loss:{:.5f}".format(i+1,loss.float().data))
        defile.write("Iter:{:05d}  loss:{:.5f}".format(i+1,loss.float().data))
        defile.flush()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='DHH')
    parser.add_argument('--lr',default=0.0001,type=float,help='learning rate')
    parser.add_argument('--batch',default=128,type=int)
    # debug information
    parser.add_argument('--savepath',default='./models/',help='the saving path for trained models')
    parser.add_argument('--logfile',default='debug.txt',type=str)
    parser.add_argument('--bits',default=16,type=int)
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

    # random seed
    torch.cuda.set_device(0)
    torch.backends.cudnn.bencmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # debug file ==> add some head information in the file
    defile=open('./logfile/'+args.logfile,"a",newline="\n")

    # configuration information
    print('========>:Config Start!')
    print("Label smoothing regularization ===>","Use LSR, eps:"+str(args.eps)+"!" if args.smooth!=0 else "No LSR!")
    print("Adative angular margin: ===>","Use AA_Margin, m:"+str(args.m)+"!" if args.m!=1 else "No AA_Margin!")
    print("Feature regularization: ===>","Use Feature regularization: type:{} lam:{}!".format(args.regularization,args.lam) if args.regularization!=0 else "No Feature regularization!")
    print('========>:Config End!')

    use_cuda=torch.cuda.is_available()
    train()
    defile.close()
