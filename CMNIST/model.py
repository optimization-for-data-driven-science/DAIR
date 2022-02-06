from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        # self._init_weights()

    ## This _init_weights function is not necessary
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            # print(m)
            # print(type(m))
            # nn.init.xavier_normal_(m.weight)
            # nn.init.constant_(m.bias, 1 / m.bias.numel())

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1, 800)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.net = nn.Linear(in_features=784 * 3, out_features=1)


    def forward(self, x):
        logits = self.net(x.flatten(1)).flatten()
        return logits


# class LeNet5(nn.Module):

#     def __init__(self):
#         super(LeNet5, self).__init__()
        
#         self.net = nn.Sequential(            
#             nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=16, out_channels=96, kernel_size=4, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Flatten(),
#             nn.Linear(in_features=96, out_features=64),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=64, out_features=1),
#         )


#     def forward(self, x):
#         logits = self.net(x).flatten()
#         return logits