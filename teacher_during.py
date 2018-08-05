#------------eeee-------------------
from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent
from utils import trainloader,testloader
from torchvision import models

import numpy as np
import os,time
import torch
from torch.autograd import Variable

def binary_output(dataloader):
    net = Resnet18PlusLatent(48)
    net.load_state_dict(torch.load('./models/teacher/acc_epoch125.0_9740.pkl'))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output1 = torch.cuda.FloatTensor()
    full_batch_output2 = torch.cuda.FloatTensor()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs1,_,outputs2,_= net(inputs)
        full_batch_output1 = torch.cat((full_batch_output1, outputs1.data), 0)
        full_batch_output2 = torch.cat((full_batch_output2, outputs2.data), 0)
    return full_batch_output1,full_batch_output2




if os.path.exists('./thash/train_binary1')  and \
   os.path.exists('./thash/test_binary2') :
    train_binary1 = torch.load('./thash/train_binary1')
    train_binary2 = torch.load('./thash/train_binary2')

else:
    print("compute thash!-----------------------------------------------")
    train_binary1,train_binary2 = binary_output(trainloader)
    if not os.path.isdir('thash'):
        os.mkdir('thash')
    torch.save(train_binary1, './thash/train_binary1')
    torch.save(train_binary2, './thash/train_binary2')





