import numpy as np
import time
import sys

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader






class CNN1min(torch.nn.Module):
    def __init__(self, out1=32, out2=16):
        super(CNN1min, self).__init__()
        self.cnn1 = torch.nn.Conv1d(in_channels=16, out_channels=out1, kernel_size=5, padding=0)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=4, stride=4)
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)
        self.cnn2 = torch.nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=5, padding=0)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=4, stride=4)
        self.bn2 = torch.nn.BatchNorm1d(num_features=out2)



        self.fc = torch.nn.Linear(out2*1497, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.bn1(x)

        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.bn2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print('x',x[0])
        out = self.sigmoid(x)
        # print('yhat', out[0])

        return out