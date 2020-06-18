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
import metrics as met
import traintest as tt

# -------------- Options -------------------
patients = [1,6,8,9,10,11,13,15]  # patient list (1-15)
patients = [1]
train = 0                           # binary, whether to train a new model
test = 1                            # binary, whether to test the model
use_existing_results = 1            # determines whether existing results should be gathered

show_auc = 1
show_bss = 1

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


def train_model(untrained_model, dataset):
    batches_per_epoch = dataset.len / batch_size
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    out_model = tt.train(model=untrained_model, train_loader=train_loader, criterion=criterion, optimizer=optimizer,
                                 n_epochs=1, batches_per_epoch=batches_per_epoch)
    torch.save(out_model, model_path)


def load_model(path):
    if os.path.isfile(path):
        out_model = torch.load(path)
        print('Model loaded')

    else:
        answer = input('Model not found, train the model?').upper()
        print(answer)
        if answer in ['Y', 'YES', ' Y', ' YES']:
            print('Training')
            train_model(batch_size, model)
            out_model = torch.load(path)
        else:
            print('Not training')
            sys.exit(0)
    return out_model


def load_or_calculate_forecasts(pt, t_model, dataset):
    dir_path = '/media/projects/daniel_lstm/results_log/' + model_name
    file_path = dir_path + '/%d.pt' % pt

    if use_existing_results:
        if os.path.isfile(file_path):
            y, yhat = torch.load(file_path)
        else:
            print('File not found for ', model_name)
            sys.exit(0)
    else:
        validation_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        print('Testing')
        y, yhat = tt.test(t_model, validation_loader)
        print('\n')

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save([y, yhat], file_path)

    return y, yhat


for pt in patients:
    print('\n---------------- Patient %d ------------------' % pt)

    # Model
    model = models.CNN1min(out1=c1, out2=c2)
    print(model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_path = "/media/projects/daniel_lstm/models/" + model_name + "_%d.pt" % pt

    # Train model
    if train:
        train_model(batch_size, model, gd.BalancedData1m(pt=pt))

    # Load model
    trained_model = load_model(model_path)

    if test:
        y, yhat = load_or_calculate_forecasts(pt, trained_model, gd.BalancedData1m(pt=pt, train=False))

        sz_yhat, inter_yhat = met.split_yhat(y, yhat)


        print('-------RESULTS-------')
        if show_auc:
            a, lo, hi = met.auc_hanleyci(sz_yhat, inter_yhat)
            print('AUC: %0.3f (%0.3f:%0.3f)' % (a, lo, hi))
            # a_s, a_e= met.auc_on_shuffled(1000, sz_yhat, inter_yhat)
            # print('Shuffled auc %0.3f (%0.4f)' % (a_s, a_e))

        if show_bss:
            bs, bs_ref, bss, bss_se = met.brier_skill_score(sz_yhat, inter_yhat, p_sz=0.5)
            print('Brier: %.3g, Bss: %0.3g (%.3g) Ref: %.3g' % (bs, bss, bss_se, bs_ref))