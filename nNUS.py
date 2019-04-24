import torch
import torchvision

from PIL import Image
import numpy as np
from torch.utils import data
import math

BATCH=128
#------------------------------------------
ftxt=open("trainpath.txt",'r')

train=[i.split(' label') for i in ftxt]
train=list(map(lambda x:[x[0],list(map(lambda y:int(y),list(x[1].replace('\n',''))))],train))#[0:50000]
ftxt.close()

ftxt=open("testpath.txt",'r')

test=[i.split(' label') for i in ftxt]
test=list(map(lambda x:[x[0],list(map(lambda y:int(y),list(x[1].replace('\n',''))))],test))#[0:10000]
ftxt.close()
#------------------------------------------

#train+=test
print("train set size:_{}".format(len(train)))
print("test set size:_{}".format(len(test)))

root=''
transform=torchvision.transforms.Compose([
#torchvision.transforms.Resize((64,64)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=[0.4388,0.4230,0.3933],std=[0.2900,0.2800,0.2967]),
])

tran=torchvision.transforms.Compose([
#torchvision.transforms.Resize((64,64)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(mean=[0.4388,0.4230,0.3933],std=[0.2900,0.2800,0.2967]),
])


class NUS(data.Dataset):
    def __init__(self,root,pathes,transforms=None):
        self.root=root
        self.pathes=pathes
        self.trans=transform

    def __getitem__(self,index):
        img_path=self.pathes[index][0]
        img_label=np.array(self.pathes[index][1]) # batchsize x 21
        img=Image.open(img_path)
        img=self.trans(img)
        return img,torch.from_numpy(img_label)

    def __len__(self):
        return len(self.pathes)

trainset=NUS(root,train,transform)
testset=NUS(root,test,transform)

trainloader=data.DataLoader(trainset,batch_size=BATCH,shuffle=False,num_workers=4)
testloader=data.DataLoader(testset,batch_size=BATCH,shuffle=False,num_workers=4)

