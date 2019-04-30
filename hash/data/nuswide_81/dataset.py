import torch
import torchvision

from PIL import Image
import numpy as np
from torch.utils import data
import math

#--------------- for training images
class Train(data.Dataset):
    def __init__(self,root,batchsize=128,transform=None):
        self.root=root
        self.transform=transform
        self.batchsize=batchsize

        # load txt file
        ftxt=open(self.root,'r')
        train=[i.split('jpg') for i in ftxt]
        train=list(map(lambda x:[x[0]+'jpg',list(map(lambda y:int(y),list(x[1].replace('\n','').replace(" ",''))))],train))
        ftxt.close()
        self.pathes=train

    def __getitem__(self,index):
        img_path=self.pathes[index][0]
        img_label=np.array(self.pathes[index][1]) # batchsize x 21
        img=Image.open(img_path)
        if self.transform:
            img=self.transform(img)
        return img,torch.from_numpy(img_label)

#--------------- for test images
class Test(data.Dataset):
    def __init__(self,root,batchsize=128,transform=None):
        self.root=root
        self.transform=transform
        self.batchsize=batchsize

        # load txt file
        ftxt=open(self.root,'r')
        train=[i.split('jpg') for i in ftxt]
        train=list(map(lambda x:[x[0]+'jpg',list(map(lambda y:int(y),list(x[1].replace('\n','').replace(" ",''))))],train))
        ftxt.close()
        self.pathes=train

    def __getitem__(self,index):
        img_path=self.pathes[index][0]
        img_label=np.array(self.pathes[index][1]) # batchsize x 21
        img=Image.open(img_path)
        if self.transform:
            img=self.transform(img)
        return img,torch.from_numpy(img_label)

#--------------- for database images
class Database(data.Dataset):
    def __init__(self,root,batchsize=128,transform=None):
        self.root=root
        self.transform=transform
        self.batchsize=batchsize

        # load txt file
        ftxt=open(self.root,'r')
        train=[i.split('jpg') for i in ftxt]
        train=list(map(lambda x:[x[0]+'jpg',list(map(lambda y:int(y),list(x[1].replace('\n','').replace(" ",''))))],train))
        ftxt.close()
        self.pathes=train

    def __getitem__(self,index):
        img_path=self.pathes[index][0]
        img_label=np.array(self.pathes[index][1]) # batchsize x 21
        img=Image.open(img_path)
        if self.transform:
            img=self.transform(img)
        return img,torch.from_numpy(img_label)
