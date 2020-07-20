import numpy as np
import time
import sys, os

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

def load_model(path):
    if os.path.isfile(path):
        out_model = torch.load(path)
        print('Model loaded from: ', path)

    # else:
        # answer = input('Model not found, train the model?').upper()
        # print(answer)
        # if answer in ['Y', 'YES', ' Y', ' YES']:
        #     print('Training')
        #     train_model(batch_size, model)
        #     out_model = torch.load(path)
    else:
        print('Model failed to load: ', path)
        sys.exit(0)
    return out_model

class CNN1min_fspec(torch.nn.Module):
    def __init__(self, out1=16, out2=32, out3=64, out4 = 32, out5 =16):
        super(CNN1min_fspec, self).__init__()

        self.cnn1 = torch.nn.Conv1d(in_channels=1920
                                    , out_channels=out1, kernel_size=3, padding=0) # 118
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2) # 59
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)


        self.cnn2 = torch.nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=3, padding=0) #57
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2) #28
        self.bn2 = torch.nn.BatchNorm1d(num_features=out2)

        self.cnn3 = torch.nn.Conv1d(in_channels=out2, out_channels=out3, kernel_size=3, padding=0) #26
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2) # 13
        self.bn3 = torch.nn.BatchNorm1d(num_features=out3)


        #23960
        #
        #1496


        self.fc1 = torch.nn.Linear(out3*13, out4) # 1496 - kernal 5 and pool 4
        self.bn4 = torch.nn.BatchNorm1d(num_features=out4)
        self.fc2 = torch.nn.Linear(out4, out5)
        self.bn5 = torch.nn.BatchNorm1d(num_features=out5)
        self.fc3 = torch.nn.Linear(out5, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # print('In Forward x require grad? ', x.requires_grad)
        # print('X shape', x.detach().numpy().shape)
        # print('X type', x.detach().type())
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

        x = self.cnn3(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.bn3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn4(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn5(x)

        x = self.fc3(x)

        # print('x',x[0])
        out = self.sigmoid(x)
        # print('yhat', out[0])

        return out

class CNN1min_spec(torch.nn.Module):
    def __init__(self, out1=16, out2=32, out3=64, out4 = 32, out5 =16):
        super(CNN1min_spec, self).__init__()

        self.cnn1 = torch.nn.Conv2d(in_channels=16, out_channels=out1, kernel_size=3, padding=0) # 118
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 59
        self.bn1 = torch.nn.BatchNorm2d(num_features=out1)


        self.cnn2 = torch.nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=3, padding=0) #57
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2) #28
        self.bn2 = torch.nn.BatchNorm2d(num_features=out2)

        self.cnn3 = torch.nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=3, padding=0) #26
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 13
        self.bn3 = torch.nn.BatchNorm2d(num_features=out3)


        #23960
        #
        #1496


        self.fc1 = torch.nn.Linear(out3*13*13, out4) # 1496 - kernal 5 and pool 4
        self.bn4 = torch.nn.BatchNorm1d(num_features=out4)
        self.fc2 = torch.nn.Linear(out4, out5)
        self.bn5 = torch.nn.BatchNorm1d(num_features=out5)
        self.fc3 = torch.nn.Linear(out5, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # print('In Forward x require grad? ', x.requires_grad)
        # print('X shape', x.detach().numpy().shape)
        # print('X type', x.detach().type())
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

        x = self.cnn3(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.bn3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn4(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn5(x)

        x = self.fc3(x)

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

class Short(torch.nn.Module):

    def __init__(self, min_model_path, out1=16, transform=None):
        super(Short, self).__init__()

        self.min_model_path = min_model_path
        self.min_model = load_model(min_model_path)
        self.out1=out1
        self.transform=transform

        self.fc1 = torch.nn.Linear(10, out1)  # 1496 - kernal 5 and pool 4
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)
        # self.fc2 = torch.nn.Linear(out3, out4)
        # self.bn4 = torch.nn.BatchNorm1d(num_features=out4)
        self.fc2 = torch.nn.Linear(out1, 1)

        self.sigmoid = torch.nn.Sigmoid()

        # self.rnn1 =

    def forward(self, x):
        print('X shape in Short start ', x.detach().numpy().shape)
        samples = x.detach().numpy().shape[0]

        with torch.no_grad():
            x = x[:, :, :239600]
            x = x.view((samples, 16, 10, 23960))
            x = x.transpose(1,2)
            x = x.reshape((samples*10,16,23960))
            if self.transform:
                # print(x.size)
                x_ = torch.empty((samples*10,16, 120, 120), dtype=torch.float)

                for sample in range(samples*10):
                    x_[sample] = self.transform(x[sample])
                x=x_
            x = self.min_model(x)
            x = x.view(samples, 10)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        out = self.sigmoid(x)


        return out