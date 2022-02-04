from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn
import torchvision.datasets.utils as dataset_utils

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

from mdset import *

from PIL import Image

import os

import numpy as np

import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))
    
    def __len__(self):
        return self.num_samples

def loadData(args):


    # transform_augment_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor()])
    transform = T.Compose([T.ToTensor()])

    MNIST_train = RMNIST('./dataset', train=True, transform=T.ToTensor(), download=True)

    MNIST_test = RMNIST('./dataset', train=False, transform=T.ToTensor(), download=True)

    # MNIST_train = ColoredMNIST(root='./dataset', env=[90, 270], train=True, transform=T.ToTensor())
    # MNIST_test = ColoredMNIST(root='./dataset', env=[0, 180], train=False, transform=T.ToTensor())

    print(len(MNIST_train))
    print(len(MNIST_test))

    # MNIST_train = ColoredMNIST(root='./dataset', env='train1')
    # MNIST_test = ColoredMNIST(root='./dataset', env='test')

    loader_train = DataLoader(MNIST_train, batch_size=args.batch_size, shuffle=True)
    
    loader_test = DataLoader(MNIST_test, batch_size=args.batch_size)

    return loader_train, loader_test