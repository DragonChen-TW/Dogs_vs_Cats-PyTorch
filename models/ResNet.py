from torch import nn
from torchvision import models
#
import sys
sys.path.append('..')
from helper import set_param_requires_grad

def ResNet():
    resnet = models.resnet18(pretrained=True)
    set_param_requires_grad(resnet)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, 2)

    return resnet
