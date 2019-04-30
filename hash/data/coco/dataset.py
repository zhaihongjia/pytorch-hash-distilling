import torch
import torchvision

from PIL import Image
import numpy as np
from torch.utils import data
import math

BATCH=12

#--------------- for training images
class Train(data.Dataset):
    def __init__(self,root,transform=None):
        self.root=root
        self.transform=transform

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

    def __len__(self):
        return len(self.pathes)

trans_train=torchvision.transforms.Compose([
torchvision.transforms.Resize((256,256)),
torchvision.transforms.RandomCrop((224,224)),
torchvision.transforms.ToTensor(),
#torchvision.transforms.Normalize(mean=[],std=[]),
])

trainset=Train("./data/coco/train.txt",trans_train)
trainloader=data.DataLoader(trainset,batch_size=BATCH,shuffle=True,num_workers=4)


#--------------- for test images
class Test(data.Dataset):
    def __init__(self,root,transform=None):
        self.root=root
        self.transform=transform

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


trans_test=torchvision.transforms.Compose([
torchvision.transforms.ToTensor(),
#torchvision.transforms.Normalize(mean=[],std=[]),
])

testset=Test("./data/coco/test.txt",trans_test)
testloader=data.DataLoader(testset,batch_size=BATCH,shuffle=False,num_workers=4)


#--------------- for database images
class Database(data.Dataset):
    def __init__(self,root,transform=None):
        self.root=root
        self.transform=transform

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

database=Database("./data/coco/database.txt",trans_test)
databaseloader=data.DataLoader(database,batch_size=BATCH,shuffle=False,num_workers=4)
