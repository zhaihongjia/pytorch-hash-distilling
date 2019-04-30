# generate the txt file

# for coco2014 imagenet nuswide_81 dataset
coco="./data/coco/"
imagenet="./data/imagenet/"
nus="./data/nuswide_81/"

# # transform txt file
# for target_root in {coco,imagenet}:
#     for filename in {"database.txt"}:
#         path=target_root+filename
#         fin=open(path,'r')
#         fout=open(target_root+'new'+filename,'w')
#         for i in fin:
#             fout.write(i.replace("/home/caozhangjie/run-czj/dataset","./data"))
#         fin.close()
#         fout.close()


# compute dataset mean and std
from data.coco.dataset import trainloader
import torch
imgs=torch.cuda.FloatTensor()
for i,j in trainloader:
    img=torch.cat((imgs,i),0)
R=imgs[:,0,:,:]
G=imgs[:,1,:,:]
B=imgs[:,2,:,:]
print("mean:{}  {}  {}.".format(R.mean(),G.mean(),B.mean()))
print("std:{}  {}  {}.".format(R.std(),G.std(),B.std()))
