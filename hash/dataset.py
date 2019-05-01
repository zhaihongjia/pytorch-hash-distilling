import torch
import torchvision
from PIL import Image
import numpy as np
from torch.utils import data
import math

class DataSet(data.Dataset):
    def __init__(self,root,postfix="jpg",transform=None):
        self.root=root
        self.transform=transform
        self.postfix=postfix

        # load txt file
        ftxt=open(self.root,'r')
        train=[i.split(self.postfix) for i in ftxt]
        train=list(map(lambda x:[x[0]+self.postfix,list(map(lambda y:int(y),list(x[1].replace('\n','').replace(" ",''))))],train))
        ftxt.close()
        self.pathes=train

    def __getitem__(self,index):
        img_path=self.pathes[index][0]
        img_label=np.array(self.pathes[index][1])
        img=Image.open(img_path)
        if self.transform:
            img=self.transform(img)
        return img,torch.from_numpy(img_label)

    def __len__(self):
        return len(self.pathes)

def load_data(mode,name,batchsize):
    if mode=='train':
        trans=torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.RandomCrop((224,224)),
        torchvision.transforms.ToTensor(),
        ])
    else:
        trans=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        ])
    if name=='coco':
        dataset=DataSet("./data/coco/"+mode+'.txt',"jpg",trans)
        dataloader=data.DataLoader(dataset,batch_size=batchsize,shuffle=True,num_workers=4)
        return dataloader
    elif name=='imagenet':
        dataset=DataSet(root,"JPEG",trans)
        dataloader=data.DataLoader(dataset,batch_size=batchsize,shuffle=True,num_workers=4)
        return dataloader
    elif name=='nus':
        dataset=DataSet(root,"jpg",trans)
        dataloader=data.DataLoader(dataset,batch_size=batchsize,shuffle=True,num_workers=4)
        return dataloader
