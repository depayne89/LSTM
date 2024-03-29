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

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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
        # print('X shape in forward', x.detach().numpy().shape)
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
        # print('testtest')
        # x = self.bn4(x)
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.bn5(x)

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
        # print('X shape', x.detach().numpy().shape)
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

    def __init__(self, min_model_path, rn1=16, out1=16, transform=None, lookBack = 1):
        super(Short, self).__init__()

        self.min_model_path = min_model_path
        min_model = load_model(min_model_path)
        self.rn1=rn1
        self.out1=out1
        self.transform=transform
        self.lookBack=lookBack
        self.num_features=32

        # remove first fc layer after CNNs
        # min_model.fc1 = Identity()
        # min_model.bn4 = Identity()

        # remove second fc layer after CNNs
        min_model.fc2 = Identity()
        min_model.bn5 = Identity()

        # remove last layer
        min_model.sigmoid = Identity()
        min_model.fc3 = Identity()

        self.min_model = min_model

        # self.rnn1 = torch.nn.LSTM(1, 16, 1, batch_first=True) # first number is input size = 1? sequence length is 10
        # (batch_size, sequence_length, number_features)
        self.rnn1 = torch.nn.LSTM(self.num_features, rn1, 1, batch_first=True)
        self.fc1 = torch.nn.Linear(rn1, out1)  # 1496 - kernal 5 and pool 4
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)
        # self.fc2 = torch.nn.Linear(out3, out4)
        # self.bn4 = torch.nn.BatchNorm1d(num_features=out4)
        self.fc2 = torch.nn.Linear(out1, 1)

        self.sigmoid = torch.nn.Sigmoid()

        # self.rnn1 =

        # rnn = nn.LSTM(10, 20, 2) # input size, hidden_size, layers
        # input = torch.randn(5, 3, 10)  seq_len, batch, input_size
        # h0 = torch.randn(2, 3, 20) num_layers * num_directions, batch, hidden_size
        # c0 = torch.randn(2, 3, 20)
        # output, (hn, cn) = rnn(input, (h0, c0))

    def forward(self, x, h0=None, c0=None, save_hidden = False):

        batch_size = x.detach().numpy().shape[0]
        sequence_length = 10*self.lookBack
        num_features = self.num_features
        input_channels = 16
        sample_trim = 239600*self.lookBack

        if h0 is not None:
            self.h0 = h0
            self.c0 = c0
        else:
            self.h0 = torch.randn(1, batch_size, self.rn1)
            self.c0 = torch.randn(1, batch_size, self.rn1)

        h0_np = self.h0.detach().numpy()
        # print('h0 shape', h0_np)

        with torch.no_grad():
            # print('x at start', x.detach().numpy().shape)
            x = x[:, :, :sample_trim] # trim to reliable number
            # print('x after trim', x.detach().numpy().shape)

            x = x.view((batch_size, input_channels, sequence_length, int(sample_trim/sequence_length)))
            x = x.transpose(1,2) # (batch_size, seq_length, channels, t)
            x = x.reshape((batch_size*sequence_length, input_channels, int(sample_trim/sequence_length)))
            # print('x before transform', x.detach().numpy().shape)

            if self.transform:
                # print(x.size)
                x_ = torch.empty((batch_size*sequence_length,input_channels, 120, 120), dtype=torch.float)

                for sample in range(batch_size*sequence_length):
                    x_[sample] = self.transform(x[sample])
                x = x_
            # print('x at after transform', x.detach().numpy().shape)

            x = self.min_model(x) # (batch_size*seq_length, 1)

            # print('x after minmodel', x.detach().numpy().shape)

            x = x.view(batch_size, sequence_length, num_features)  # (Batch_size, seq_length)

        # x = x.view((batch_size, sequence_length, num_features))  # (batch_size, seq_length, num_features)
        x, (h1, c1) = self.rnn1(x, (self.h0, self.c0))  # (batch_size, seq_length, hidden_size)
        # x=torch.relu(x)
        x = x.reshape((batch_size*sequence_length, self.rn1))# (batch_size * seq_length, hidden_size)

        # x = x.view((batch_size*sequence_length, self.rn1))  # (batch_size * seq_length, hidden_size)

        x = self.fc1(x)
        x = torch.relu(x)


        x = self.bn1(x)

        x = self.fc2(x)

        out = self.sigmoid(x)  # (batch_size * seq_length, hidden_size)
        out = out.reshape((batch_size, sequence_length, 1))# (batch_size * seq_length, hidden_size)

        out = out[:, -1, :]

        h1_np = h1.detach().numpy()
        # print('h1 shape', h1_np)

        if save_hidden:
            return out, h1, c1
        else:
            return out


class Medium(torch.nn.Module):

    def __init__(self, min_model_path, rn1=16, out1=16, transform=None, hrsBack=24):
        super(Medium, self).__init__()

        self.min_model_path = min_model_path
        min_model = load_model(min_model_path)
        self.rn1=rn1
        self.out1=out1
        self.transform=transform
        self.hrsBack=hrsBack
        self.num_features = 32

        # remove first fc layer after CNNs
        # min_model.fc1 = Identity()
        # min_model.bn4 = Identity()

        # remove second fc layer after CNNs
        min_model.fc2 = Identity()
        min_model.bn5 = Identity()

        # remove last layer
        min_model.sigmoid = Identity()
        min_model.fc3 = Identity()

        self.min_model = min_model

        self.rnn1 = torch.nn.LSTM(self.num_features, 16, 1, batch_first=True)

        self.fc1 = torch.nn.Linear(rn1, out1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)

        self.fc2 = torch.nn.Linear(out1, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, h0=None, c0=None, save_hidden=False):
        batch_size = x.detach().numpy().shape[0]
        sequence_length = self.hrsBack + 1
        num_features=self.num_features
        input_channels=16
        sample_trim = 23960

        if h0 is not None:
            self.h0 = h0
            self.c0 = c0

        else:
            self.h0 = torch.randn(1, batch_size, self.rn1)
            self.c0 = torch.randn(1, batch_size, self.rn1)

        with torch.no_grad():
            # print('x shape ', x.detach().numpy().shape)
            # input shape: (batch_size, sequence_length, input_channels, sample_trim)
            x = x.reshape((batch_size*sequence_length, input_channels, sample_trim))

            if self.transform:
                x_ = torch.empty((batch_size*sequence_length, input_channels, 120, 120), dtype=torch.float)

                for sample in range(batch_size*sequence_length):
                    x_[sample] = self.transform(x[sample])
                x = x_

            x = self.min_model(x)
            x = x.view(batch_size, sequence_length, num_features)

        x, (h1, c1) = self.rnn1(x, (self.h0, self.c0))
        # x = torch.relu(x) # may remove
        x = x.reshape((batch_size*sequence_length, self.rn1))

        x = self.fc1(x)
        x = torch.relu(x)
        # print('x before batch norm', x.detach().numpy().shape)

        if not save_hidden:
            x = self.bn1(x) # Remove for batch size 1 (ie whole test set)

        x =self.fc2(x)

        out = self.sigmoid(x)
        out = out.reshape((batch_size, sequence_length, 1))

        out = out[:, -1, :]  # out[:, -1, :]

        if save_hidden:
            return out, h1, c1
        else:
            return out

class Long(torch.nn.Module):

    def __init__(self, min_model_path, rn1=16, out1=16, transform=None, daysBack=30):
        super(Long, self).__init__()

        self.min_model_path = min_model_path
        min_model = load_model(min_model_path)
        self.rn1 = rn1
        self.out1 = out1
        self.transform = transform
        self.days_back = daysBack
        self.num_features = 32  # -remove one layer: 16. remove all layers: 64*13*13

        # remove first fc layer after CNNs
        # min_model.fc1 = Identity()
        # min_model.bn4 = Identity()

        # remove second fc layer after CNNs
        min_model.fc2 = Identity()
        min_model.bn5 = Identity()

        # remove last layer
        min_model.sigmoid = Identity()
        min_model.fc3 = Identity()

        self.min_model = min_model

        self.rnn1 = torch.nn.LSTM(self.num_features, 16, 1, batch_first=True)

        self.fc1 = torch.nn.Linear(rn1, out1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)

        self.fc2 = torch.nn.Linear(out1, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, h0=None, c0=None, save_hidden=False):

        batch_size = x.detach().numpy().shape[0]
        sequence_length = self.days_back + 1
        num_features = self.num_features
        input_channels = 16
        sample_trim = 23960


        if h0 is not None:
            self.h0 = h0
            self.c0 = c0
        else:
            self.h0 = torch.randn(1, batch_size, self.rn1)
            self.c0 = torch.randn(1, batch_size, self.rn1)

        with torch.no_grad():
            # print('x shape ', x.detach().numpy().shape)
            x = x.reshape((batch_size*sequence_length, input_channels, sample_trim))

            if self.transform:
                x_ = torch.empty((batch_size*sequence_length, input_channels, 120, 120), dtype=torch.float)

                for sample in range(batch_size*sequence_length):
                    x_[sample] = self.transform(x[sample])
                x = x_

            x = self.min_model(x)
            x = x.view((batch_size, sequence_length, num_features))

        x, (h1, c1) = self.rnn1(x, (self.h0, self.c0))

        x = x.reshape((batch_size*sequence_length, self.rn1))

        x = self.fc1(x)
        x = torch.relu(x)
        if not save_hidden:
            x = self.bn1(x)  # Remove for batch size 1

        x = self.fc2(x)
        out = self.sigmoid(x)
        out = out.reshape((batch_size, sequence_length, 1))

        out = out[:, -1, :]  # out[:, -1, :]

        if save_hidden:
            return out, h1, c1
        else:
            return out


class Combo(torch.nn.Module):

    def __init__(self, min_model_path, short_model_path, med_model_path, long_model_path,
                 out1=16, transform=None):
        super(Combo, self).__init__()

        self.min_model_path = min_model_path

        self.short_model = load_model(short_model_path)
        self.med_model = load_model(med_model_path)
        self.long_model = load_model(long_model_path)

        self.out1 = out1

        self.fc1 = torch.nn.Linear(3, out1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=out1)
        self.fc2 = torch.nn.Linear(out1, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        # batch_size = x['short'].shape[0]
        # print('GOT TO FORWARD')
        with torch.no_grad():


            x_short = self.short_model(x['short'])
            x_med = self.med_model(x['medium'])
            x_long = self.long_model(x['long'])



            x =torch.cat((x_short, x_med, x_long), 1)

        print('x shape before combo', x.detach().numpy().shape)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        out = self.sigmoid(x)
        return out


