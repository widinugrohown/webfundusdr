import torch.nn as nn
from torchvision import models

class model(nn.Module):
    def __init__(self, num_class=5):
        super().__init__()
        googlenet = models.googlenet(pretrained=True)
        modules = list(googlenet.children())[:-1] 
        self.googlenet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(1024, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        out = self.googlenet(x)
        out = out.reshape(out.size(0), -1)
        out = self.bn(self.fc1(out))
        return out