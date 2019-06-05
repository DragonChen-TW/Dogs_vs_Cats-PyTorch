from torch import nn
from torchvision import models
#
import sys
sys.path.append('..')
from helper import set_param_requires_grad

def AlexNet():
    alexnet = models.alexnet(pretrained=True)
    set_param_requires_grad(alexnet)
    alexnet.classifier[6] = nn.Linear(4096, 2)

    return alexnet
