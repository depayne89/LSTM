import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import time
import sys, os

import matplotlib.pyplot as plt

import create_datasets as cd
import models
import get_datasets as gd
import validate as val

# -------------- Settings -------------------
patients = [1,6,8,9,10,11,13,15]  # patient list (1-15)
patients = [1]
train = 1                           # binary, whether to train a new model
test = 1                            # binary, whether to test the model


# ----------- Hyperparameters ---------------

version=1

n_epochs = 1
batch_size = 20
learning_rate = 0.001
c1=32
c2=16

model_name = '1m_v%d_c%dp4c%dp4d_adam' % (version, c1, c2) + str(learning_rate)[2:]
print(model_name)
# Name guide in order
"""
Timescale: 1m/short/medium/long
version: iteration of this timescale
architechture: c% - convlayer with % filters, p4 - maxpool from 4 values, d - dense/fc layer
optimizer: name (eg adam) followed by learning rate (with 0. removed, eg 0.001 -> 001
patient: added to the end on save
"""


def train_model(pt, batch_size, model):
    train_dataset = gd.BalancedData1m(pt=pt)
    batches_per_epoch = train_dataset.len / batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    model = models.train(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer,
                                 n_epochs=1, batches_per_epoch=batches_per_epoch)
    torch.save(model, model_path)

for pt in patients:
    print('\n---------------- Patient %d ------------------' % pt)

    # Model
    model = models.CNN1min(out1=c1, out2=c2)
    print(model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_path = "/media/projects/daniel_lstm/models/" + model_name + "_%d.pt" % pt

    # Train
    if train:
        train_model(pt, batch_size, model)

    # Load model
    if os.path.isfile(model_path):
        trained_model = torch.load(model_path)
        print('Model loaded')

    else:
        answer = input('Model not found, train the model?').upper()
        print(answer)
        if answer in ['Y', 'YES', ' Y', ' YES']:
            print('Training')
            train_model(pt, batch_size, model)
            trained_model = torch.load(model_path)
        else:
            print('Not training')
            sys.exit(0)

    if test:

        test_dataset = gd.BalancedData1m(pt=pt, train=False)
        validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        print('Testing')
        y, yhat = models.test(trained_model, validation_loader)
        # torch.save([y, yhat], )
        print('Yhat', yhat)
        sz_yhat, inter_yhat = val.split_yhat(y, yhat)
        print('\n')

        print('-------RESULTS-------')
        a, lo, hi = val.auc_hanleyci(sz_yhat, inter_yhat)