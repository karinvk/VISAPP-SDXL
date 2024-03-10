import torch
import torchvision
from torch import nn
import torch.nn.functional as func
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes, input_img_size):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.input_img_size = input_img_size

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes, input_img_size):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.input_img_size = input_img_size

    def forward(self, x):
        return self.model(x)
