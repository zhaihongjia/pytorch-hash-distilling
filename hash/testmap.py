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
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()

    query_times = tst_binary.shape[0]    #testdataset
    trainset_len = train_binary.shape[0]  #traindataset
    AP = np.zeros(query_times)                #查询次数---就是testdataset的数量---对于每一个测试图片进行检索
    Ns = np.arange(1, trainset_len + 1)       #---准确率的分母计算---- 1,2,3,--tranlen

    total_time_start = time.time()
    for i in range(query_times):
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        #---------hash code 一样的数目    数组
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)  #don't need to divide binary length
        #print("query_result",query_result)
        sort_indices = np.argsort(query_result)  #将元素从小到大排序，并返回index
        buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)   #----训练数据中和当前测试i图片一样label的数组表示
        P = np.cumsum(buffer_yes) / Ns  #------precision
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
    print('total query time = ', time.time() - total_time_start)
    map = np.mean(AP)
    file1.write("     map: "+str(map)+"\n")
    print("map",map)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Test map of the retrieval model!")
    parser.add_argument('--batchsize',default=128,,type=int)
    parser.add_argument('--dataset',default='nus',type=str)
    parser.add_argument('--path',default='./',type=str)
    parger.add_argument('--class_num',default=21,type=int)
    args=parser.parse_args()

    for bits in　{16,32,48,64}:
        target_root=args.path
        filename=target_root+"map.txt"
        file1=open(filename,"w",newline='\n')
        models=os.listdir(target_root)
        modelspath=[os.path.join(target_root,model) for model in models if model.endswith('pkl')]
        for modelpath in modelspath:
            #------judge bits-----------
            if "bit12" in modelpath:
                bits=12
            elif "bit24" in modelpath:
                bits=24
            elif "bit36" in modelpath:
                bits=36
            elif "bit48" in modelpath:
                bits=48

            net=AlexNet(bits,args.class_num)
            net.load_state_dict(torch.load(modelpath))

            #-------compute map value
            testloader=load_data(args.dataset,'jpg',args.batchsize)
            databaseloader=load_data(args.dataset,'jpg',args.batchsize)
            test_bin,test_label=binary_output(testloader,net)
            base_bin,base_label=binary_output(databaseloader,net)
            file1.write(modelpath)
            file1.flush()
            print(modelpath)
            precision(base_bin,base_label,test_bin,test_label)
        file1.close()
