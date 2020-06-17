import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_printoptions(linewidth=120)

import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose(
    [transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)

import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        #super(Network, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        ## implement forward pass
        t = t
        
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        t = self.fc2(t)
        t = F.relu(t)
        
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        
        return t

network = Network()
network

# torch.set_grad_enabled(False)

sample = next(iter(train_set))

pred = network(sample[0].unsqueeze(0))