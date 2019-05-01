# generate the txt file

# for coco2014 imagenet nuswide_81 dataset
coco="./data/coco/"
imagenet="./data/imagenet/"
nus="./data/nuswide_81/"

# compute dataset mean and std
from dataset import load_data
import torch
trainloader=load_data('./data/nuswide_81/test.txt','nus',256)
imgs=torch.FloatTensor()

for i,j in trainloader:
	#i=i.cuda()
	#imgs=torch.cat((imgs,i),0)
	print('load!')

# R=imgs[:,0,:,:]
# G=imgs[:,1,:,:]
# B=imgs[:,2,:,:]
# print("mean:{}  {}  {}.".format(R.mean(),G.mean(),B.mean()))
# print("std:{}  {}  {}.".format(R.std(),G.std(),B.std()))
