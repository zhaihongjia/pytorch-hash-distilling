# for visualization of the performance of the retrieval model
# evaluation of map
from CIFAR10 import trainloader,testloader
from torchvision import models
import torchvision
from net.Network import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os,argparse
import numpy as np
import os,time
import torch
from dataset import load_data
from torch.autograd import Variable
toimg=torchvision.transforms.ToPILImage()

def binary_output(dataloader,net):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.FloatTensor()
    full_batch_label = torch.LongTensor()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(targets)
        features,outputs = net(inputs,labels)
        full_batch_output = torch.cat((full_batch_output, torch.sigmoid(features).data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.round(full_batch_output), full_batch_label

def precision(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.numpy()
    tst_binary = tst_binary.numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.numpy()


    query_times = tst_binary.shape[0]    #testdataset
    trainset_len = train_binary.shape[0]  #traindataset
    #Precision = np.zeros((query_times,trainset_len))
    #Recall= np.zeros((query_times,trainset_len))
    Precision = np.zeros((1,trainset_len))
    Recall= np.zeros((1,trainset_len))
    Ns = np.arange(1, trainset_len + 1)       #---准确率的分母计算---- 1,2,3,--tranlen

    total_time_start = time.time()
    plt.figure(1)
    for i in range(query_times):
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        #---------hash code 一样的数目    数组
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)  #don't need to divide binary length
        sort_indices = np.argsort(query_result)  #将元素从小到大排序，并返回index
        buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)   #----训练数据中和当前测试i图片一样label的数组表示
        Precision[0][:] = np.cumsum(buffer_yes) / Ns  #------precision
        #AP[i] = np.sum(P) /sum(buffer_yes)
        Recall[0][:]= np.cumsum(buffer_yes) / sum(buffer_yes)
        #plt.plot(Recall[0],Precision[0])
        np.save("yes.npy",buffer_yes)
    #plot data
    plt.plot(Recall[0]*buffer_yes,Precision[0]*buffer_yes)
    plt.show()
    np.save("pre.npy",Precision[0])
    np.save("rec.npy",Recall[0])

def plot_pr_curve(p,r,yes):
    sort_index=np.argsort(r)
    p=np.array(p[sort_index])
    r=np.array(r[sort_index])
    rnew=np.linspace(r.min(),r.max(),300)
    #f=interp1d(r,p,kind="cubic")
    plt.plot(r,p,linewidth=2)
    #pnew=f(rnew)
    #plt.plot(rnew,pnew)
    plt.show()

def Top_N_pre(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()

    query_times = tst_binary.shape[0]    #testdataset
    trainset_len = train_binary.shape[0]  #traindataset
    p=np.zeros(trainset_len)
    Ns = np.arange(1, trainset_len + 1)       #---准确率的分母计算---- 1,2,3,--tranlen

    total_time_start = time.time()
    for i in range(query_times):
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        #---------hash code 一样的数目    数组
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)  #don't need to divide binary length
        sort_indices = np.argsort(query_result)  #将元素从小到大排序，并返回index
        buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)   #----训练数据中和当前测试i图片一样label的数组表示
        p += np.cumsum(buffer_yes) / Ns  #------precision
    p=p/query_times
    plt.plot(p)
    plt.show()
    return p

# hamming radius=2
def hamming_r_2(train_bin,train_lab,test_bin,test_label):
    train_bin=train_bin.cpu().numpy()
    train_bin=np.asarray(train_bin,np.int32)
    train_label=train_label.cpu().numpy()
    test_bin=test_bin.cpu().numpy()
    test_bin=np.asarray(test_bin,np.int32)
    test_label=test_label.cpu().numpy()

    query_times=test_label.size()[0]
    pre=np.zeros(query_times)

    for i in range(query_times):
        query_label=test_label[i,:]
        query_bin=test_bin[i,:]

        query_result=np.count_nozero(query_bin!=train_bin,axis=1)
        sort_indices=np.argwhere(query_result<3)
        query_yes=(np.asarray((query_label==1)&(train_label[sort_indices]==1)).astype(int).sum(axis=1)>=1).astype(int)
        pre[i]=sum(query_yes)/query_yes.size

    return np.mean(pre)


if __name__=="__main__":
    train_label=torch.load("train_label")
    train_binary=torch.load("train_binary")
    test_label=torch.load("test_label")
    test_binary=torch.load("test_binary")
    #precision(train_binary, train_label, test_binary, test_label)

    Precision=np.load("pre.npy")
    Recall=np.load("rec.npy")
    yes=np.load("yes.npy")
    #plot_pr_curve(Precision,Recall,yes)
    Top_N_pre(train_binary, train_label, test_binary, test_label)
