import numpy as np
import time
import sys

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader




class CNN1min_spec(torch.nn.Module):
    def __init__(self, out1=32, out2=16, out3=16, out4 = 16):
        super(CNN1min, self).__init__()

        self.cnn1 = torch.nn.Conv1d(in_channels=16, out_channels=out1, kernel_size=5, padding=0) # 23956
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=4, stride=4) #5989
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)


        self.cnn2 = torch.nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=5, padding=0, dilation=2) #5981
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=4, stride=4) #1495
        self.bn2 = torch.nn.BatchNorm1d(num_features=out2)

        # self.cnn3 = torch.nn.Conv1d(in_channels=out2, out_channels=out3, kernel_size=5, padding=0)
        # self.maxpool3 = torch.nn.MaxPool1d(kernel_size=4, stride=4)
        # self.bn3 = torch.nn.BatchNorm1d(num_features=out3)


        #23960
        #
        #1496


        self.fc1 = torch.nn.Linear(out2*1495, out3) # 1496 - kernal 5 and pool 4
        self.bn3 = torch.nn.BatchNorm1d(num_features=out3)
        # self.fc2 = torch.nn.Linear(out3, out4)
        # self.bn4 = torch.nn.BatchNorm1d(num_features=out4)
        self.fc2 = torch.nn.Linear(out3, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # print('In Forward x require grad? ', x.requires_grad)
        print('X shape', x.detach().numpy().shape)
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        # print('after cnn1 x require grad? ', x.requires_grad)

        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        # print('after cnn2 x require grad? ', x.requires_grad)

        # x = self.cnn3(x)
        # x = torch.relu(x)
        # x = self.maxpool3(x)
        # x = self.bn3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn3(x)
        # x = self.fc2(x)
        # x = torch.relu(x)
        # x = self.bn4(x)

        x = self.fc2(x)

        # print('x',x[0])
        out = self.sigmoid(x)
        # print('yhat', out[0])

        return out

class CNN1min(torch.nn.Module):
    def __init__(self, out1=32, out2=16, out3=16, out4 = 16):
        super(CNN1min, self).__init__()

        self.cnn1 = torch.nn.Conv1d(in_channels=16, out_channels=out1, kernel_size=5, padding=0) # 23956
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=4, stride=4) #5989
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)


        self.cnn2 = torch.nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=5, padding=0, dilation=2) #5981
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=4, stride=4) #1495
        self.bn2 = torch.nn.BatchNorm1d(num_features=out2)

        # self.cnn3 = torch.nn.Conv1d(in_channels=out2, out_channels=out3, kernel_size=5, padding=0)
        # self.maxpool3 = torch.nn.MaxPool1d(kernel_size=4, stride=4)
        # self.bn3 = torch.nn.BatchNorm1d(num_features=out3)


        #23960
        #
        #1496


        self.fc1 = torch.nn.Linear(out2*1495, out3) # 1496 - kernal 5 and pool 4
        self.bn3 = torch.nn.BatchNorm1d(num_features=out3)
        # self.fc2 = torch.nn.Linear(out3, out4)
        # self.bn4 = torch.nn.BatchNorm1d(num_features=out4)
        self.fc2 = torch.nn.Linear(out3, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # print('In Forward x require grad? ', x.requires_grad)
        print('X shape', x.detach().numpy().shape)
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        # print('after cnn1 x require grad? ', x.requires_grad)

        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        # print('after cnn2 x require grad? ', x.requires_grad)

        # x = self.cnn3(x)
        # x = torch.relu(x)
        # x = self.maxpool3(x)
        # x = self.bn3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn3(x)
        # x = self.fc2(x)
        # x = torch.relu(x)
        # x = self.bn4(x)

        x = self.fc2(x)

        # print('x',x[0])
        out = self.sigmoid(x)
        # print('yhat', out[0])

        return out