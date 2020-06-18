import numpy as np
import time
import sys

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader



def train(model, train_loader, criterion, optimizer, n_epochs, batches_per_epoch):
    for epoch in range(n_epochs):
        t0 = time.time()
        cost = 0
        batch = 0
        for x, y in train_loader:
            batch += 1
            t = time.time() - t0
            percent_done = batch / batches_per_epoch
            sys.stdout.write('\rBatch %d of %d, %0.1f done, %0.2f of %0.2f seconds' % (
            batch, batches_per_epoch, percent_done * 100, t, t / percent_done))

            optimizer.zero_grad()  # clear gradient
            z = model(x)  # make prediciton
            loss = criterion(z, y)  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # update parameters
            cost += loss.item()

    return model


def test(trained_model, validation_loader):
    correct = 0
    batch = 0
    y_tensor = torch.empty(0, dtype=torch.float)
    yhat_tensor = torch.empty(0, dtype=torch.float)
    for x, y in validation_loader:
        batch += 1
        sys.stdout.write('\rBatch %d' % batch)

        yhat = trained_model(x)  # was z
        # print('yhat', yhat.data)
        # _, yhat = torch.max(z.data, 1.)
        y_tensor = torch.cat((y_tensor, y.type(torch.float)))
        yhat_tensor = torch.cat((yhat_tensor, yhat.type(torch.float)))
    return y_tensor.detach().numpy(), yhat_tensor.detach().numpy()


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