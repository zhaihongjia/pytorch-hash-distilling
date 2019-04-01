import torch
from torch import nn,optim
from torch.autograd import Variable
from CIFAR10_AGU import train_loader,val_loader
from model import *
import argparse

def Train(model, train_loader, val_loader, loss_function, optimizer):
    for epoch in torch.arange(1,300):
        #------------for train----------
        model.train(True)

        runing_loss=0.0
        running_acc=0.0
        total=0
        correct=0

        for (tr_input,label) in train_loader:
            tr_input=Variable(tr_input.cuda())
            label=Variable(label.cuda())
            optimizer.zero_grad()
            _,output=model(tr_input)
            loss=loss_function(output,label)
            loss.backward()
            runing_loss+=loss.data
            optimizer.step()
            prediction=torch.max(output.data,1)[1]
            total+=label.size(0)
            correct+=(prediction==label).sum()
        print("train: epoch:%d  correct:%d/%d  loss:%.4f"%(epoch,runing_loss,correct,total))

        if True:
            model.train(False)
            total=0
            correct=0
            for (y_input,y_label) in val_loader:
                y_input=Variable(y_input.cuda())
                y_label=Variable(y_label.cuda())
                _,y_output=model(y_input)
                _,y_prediction=torch.max(y_output.data,1)
                total+=y_label.size(0)
                correct+=(y_prediction==y_label).sum()
            print("test: epoch:%d  correct:%d/%d"%(epoch,correct,total))
            if epoch%args,save_iter==0:
                best=correct
                filename="models/"+args.name+"/epoch%d_%d_"%(epoch,correct)+".pkl"
                torch.save(model.state_dict(),filename)
                print('======>saving model')
        if epoch in {120,240,300}:
            LR=LR*0.1
            optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Train resnet series on cifar10!")
    parser.add_argument('--lr',default=0.01,type=float)
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--weight_decay',default=5e-5,type=float)
    parser.add_argument('--save_iter',default=10,type=int)
    parser.add_argument('--name',default="resnet34",type=str)
    args=parser.parse_args()


    for bits in {12,24,36,48}:
        LR=args.lr
        model=(bits)
        #model.load_state_dict(torch.load('models/resnet18/epoch96_8738_resnet18.pkl'))
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=self.momentum, weight_decay=self.weight_decay)
        Train(model, train_loader, val_loader, criterion, optimizer)
