from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()
        
        self.net = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=96, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=96, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=n_classes),
        )


    def forward(self, x):
        logits = self.net(x)
        return logits