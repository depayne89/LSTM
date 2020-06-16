import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import time
import sys

import matplotlib.pyplot as plt

import create_datasets as cd
import models
import get_datasets as gd

n_epochs = 1

patients = [1,3,6,8,9,10,11,13,15]
patients = [1]


for pt in patients:
    print('\n---------------- Patient %d ------------------' % pt)
    train_dataset = gd.BalancedData1m(pt=pt)
    test_dataset = gd.BalancedData1m(pt=pt, train=False)
    batches_per_epoch = train_dataset.len/20

    model = models.CNN1min(out1=32, out2=16)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train_loader = DataLoader(dataset=train_dataset, batch_size=20)
    validation_loader = DataLoader(dataset=test_dataset, batch_size=20)


    # Train

    for epoch in range(n_epochs):
        t0 = time.time()
        cost = 0
        batch = 0
        for x, y in train_loader:
            batch+=1
            t = time.time() - t0
            percent_done = batch/batches_per_epoch
            print('\n')
            sys.stdout.write('\rBatch %d of %d, %0.1f done, %0.2f of %0.2f seconds' % (batch, batches_per_epoch, percent_done*100, t, t/percent_done))

            optimizer.zero_grad() # clear gradient
            z = model(x)  # make prediciton
            loss = criterion(z, y)  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # update parameters
            cost += loss.item()

    # Test

    print('\n')
    correct=0
    batch=0
    for x, y in validation_loader:
        batch+=1
        print('\n')
        sys.stdout.write('\rBatch %d' % batch)

        z = model(x)
        _, yhat = torch.max(z.data, 1)
        correct +=(yhat==y).sum().item()

    accuracy = correct / len(test_dataset)
    print('Accuracy', accuracy)