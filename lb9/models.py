from torchvision import models
from torchvision import transforms
from torch import nn

def get_resNet18(num_cls:int):

    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights)

    for param in resnet.parameters():
        param.requires_grad = False

    resnet.fc = nn.Linear(resnet.fc.in_features, out_features=num_cls,)
    return resnet

def get_resNet50(num_cls:int):

    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights)

    for param in resnet.parameters():
        param.requires_grad = False

    resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_cls)
    return resnet