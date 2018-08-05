import torch
import torchvision.models as models

from net.Resnet import Resnet18PlusLatent
from net.Squeezenet import SqueezenetPlusLatent

t=Resnet18PlusLatent(48)
t.load_state_dict(torch.load("./models/teacher/acc_epoch125.0_9740.pkl"))
s=SqueezenetPlusLatent(48)
s.load_state_dict(torch.load("./models/student/mse_epoch40.0_611.pkl"))
print("teacher:")
for name,p in t.named_parameters():
    print(name,p)
print("student:")
for name,p in s.named_parameters():
    print(name,p)

