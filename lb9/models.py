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

def get_resNet19_transforms():
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.TrivialAugmentWide(
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.Normalize(mean, std),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std),
        transforms.ToTensor(),
    ])

    return train_transforms, test_transforms

def get_resNet50(num_cls:int):

    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights)

    for param in resnet.parameters():
        param.requires_grad = False

    resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_cls)