#importing stuff
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)
    def forward(self,x):
        F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
model=NN(784,10)
x=torch.randn(64,784)
