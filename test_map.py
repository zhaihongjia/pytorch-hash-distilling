from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent
from utils import trainloader,testloader
from torchvision import models

import numpy as np
import os,time
import torch
from torch.autograd import Variable

def binary_output(dataloader,net):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        _,outputs,_,_, = net(inputs)
        full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.round(full_batch_output), full_batch_label

def precision(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()
    # print("tst_binary size:",tst_binary)
    # print("tst_label size:",tst_label)
    # print("trn_binary size:",trn_binary)
    # print("trn_label size:",trn_label)

    query_times = tst_binary.shape[0]   #testdataset
    trainset_len = train_binary.shape[0]  #traindataset
    AP = np.zeros(query_times)                #查询次数---就是testdataset的数量---对于每一个测试图片进行检索
    Ns = np.arange(1, trainset_len + 1)       #---准确率的分母计算---- 1,2,3,--tranlen

    total_time_start = time.time()
    for i in range(query_times):
        #print('Query ', i+1)
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        #---------hash code 一样的数目    数组
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)  #don't need to divide binary length
        #print("query_result",query_result)
        sort_indices = np.argsort(query_result)  #将元素从小到大排序，并返回index
        buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)   #----训练数据中和当前测试i图片一样label的数组表示
        #print("buffer_yes:",buffer_yes)
        P = np.cumsum(buffer_yes) / Ns  #------precision
        #print("p:",P)
        #print("P * buffer_yes:",P * buffer_yes)
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)

    map = np.mean(AP)
    file1.write("     map: "+str(map)+"\n")
    print("map",map)
    print('total query time = ', time.time() - total_time_start)


bits=12
#-------get the model name in the target directory and load model---------------------
target_root="./models/student/112/"
filename=target_root+"map.txt"
file1=open(filename,"w",newline="\n")
models=os.listdir(target_root)
modelspath=[os.path.join(target_root,model) for model in models if model.endswith("pkl")]
for modelpath in modelspath:
    #net=Resnet18PlusLatent(bits)
    net=SqueezenetPlusLatent(bits)
    net.load_state_dict(torch.load(modelpath))

    #---------compute map-------------------
    train_binary, train_label = binary_output(trainloader,net)
    test_binary, test_label = binary_output(testloader,net)
    file1.write(modelpath)
    file1.flush()
    print(modelpath)
    precision(train_binary, train_label, test_binary, test_label)
file1.close()




















#----------
