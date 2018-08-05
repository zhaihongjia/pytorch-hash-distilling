import torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision import models,transforms,datasets
from torch.utils.data import DataLoader

BATCH_SIZE = 2048
LR = 0.001
DATA_PATH='./data'

def TransformData():
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  ])

    train_dataset = datasets.CIFAR10(DATA_PATH, train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(DATA_PATH, train=False, transform=transform_val, download=True)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, False)
    return train_loader,val_loader


def Train(model, train_loader, val_loader, loss_function, optimizer):
    best=0
    for epoch in torch.arange(1,35000):
        #-----------for train----------
        model.train(True)

        runing_loss=0.0
        running_acc=0.0
        total=0
        correct=0

        for (input,label) in train_loader:
            input=Variable(input.cuda())
            label=Variable(label.cuda())
            optimizer.zero_grad()
            output=model(input)
            loss=loss_function(output,label)
            loss.backward()
            runing_loss+=loss.data
            optimizer.step()
            prediction=torch.max(output.data,1)[1]
            total+=label.size(0)
            correct+=(prediction==label).sum()
        #-----每一个epoch 打印一次信息----
        print("train: epoch:%d  loss:%.4f total:%d correct:%d"%(epoch,runing_loss,total,correct))

        if True:
            model.train(False)
            total=0
            correct=0
            for (y_input,y_label) in val_loader:
                y_input=Variable(y_input.cuda())
                y_label=Variable(y_label.cuda())
                y_output=model(y_input)
                _,y_prediction=torch.max(y_output.data,1)
                total+=y_label.size(0)
                correct+=(y_prediction==y_label).sum()
            print("test: epoch:%d  total:%d  correct:%d"%(epoch,total,correct))
            if correct>best or epoch%10==0:
                best=correct
                filename="models/resnet18/epoch%d_%d_"%(epoch,correct)+"resnet18.pkl"
                torch.save(model.state_dict(),filename)
                print('saving model---------------------')

if __name__ == '__main__':
    train_loader,val_loader = TransformData()
    model=models.resnet18(pretrained=True)
    model.avgpool = nn.AvgPool2d(1, stride=1)
    model.fc = nn.Linear(in_features = 512, out_features = 10)
    model.load_state_dict(torch.load('models/resnet18/epoch96_8738_resnet18.pkl'))
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    Train(model, train_loader, val_loader, criterion, optimizer)
