import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 8, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(8, 12, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(37632, 10000),
            nn.Linear(10000, 1000),
            nn.Linear(1000, 240),
            nn.Linear(240, 84),
            nn.Linear(84, 2),
        )
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

if __name__ == '__main__':
    x = torch.rand((4, 3, 224, 224))
    model = LeNet()
    y = model(x)

    print(y.shape)
