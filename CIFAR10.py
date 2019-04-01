import torchvision
form torchvision import models,transforms,datasets
import torch

DATA_PATH='./data'
BATCH=128
transform_train = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

transform_test=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),])

train_dataset = datasets.CIFAR10(DATA_PATH, train=True, transform=transform_train, download=True)
val_dataset = datasets.CIFAR10(DATA_PATH, train=False, transform=transform_val, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, True,num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, False,num_workes=4)
