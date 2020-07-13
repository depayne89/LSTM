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
patients = [6, 8,10,11]  # patient list (1-15)
# patients = [6, 9, 10, 11]  # patient list (1-15)

# patients = [6]  # patient list (1-15)

train = 1                           # binary, whether to train a new model
test = 1                            # binary, whether to test the model
use_existing_results = 0            # determines whether existing results should be gathered
test_iterations = 1                 # odd, How many test sets to take median from

show_auc = 1
show_bss = 1

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------- Hyperparameters ---------------

version = 1

data_mult = 4  # how many interictal sample per seizure (this is balanced by including duplicate sz samples)
duplicate_ictal = False

n_epochs = 1
batch_size = 160  # for 1mBalanced: 20 samples / sz
learning_rate = .0001
c1=16  #15
c2=32  #64
# c3 =128
fc1=16  #128
# fc2 = 32


def train_model(untrained_model, dataset, test_dataset):
    batches_per_epoch = np.ceil(dataset.len / batch_size)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_dataset.len)
    out_model = tt.train(model=untrained_model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer,
                                 n_epochs=n_epochs, batches_per_epoch=batches_per_epoch, model_name=model_name + '_%d' % pt)
    print('Model saved as: ', model_path)
    torch.save(out_model, model_path)


def load_model(path):
    if os.path.isfile(path):
        out_model = torch.load(path)
        print('Model loaded from: ', path)

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

    if use_existing_results and not train:  # Always want new results if retrained
        if os.path.isfile(file_path):
            y, yhat = torch.load(file_path)
        else:
            print('File not found for ', model_name)
            sys.exit(0)
    else:
        validation_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        print('Testing')
        y, yhat = tt.test(pt, t_model, validation_loader)
        print('\n')

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save([y, yhat], file_path)

    return y, yhat


for pt in patients:
    print('\n---------------- Patient %d ------------------' % pt)

    if pt==11 or pt==13:
        learning_rate = .001

    model_name = '1m_v%d_c%dp4c%dp4d%dd_%d_adam' % (version, c1, c2, fc1, batch_size) + str(learning_rate)[2:]
    print(model_name)
    # Name guide in order
    """
    Timescale: 1m/short/medium/long
    version: iteration of this timescale
    architechture: c% - convlayer with % filters, p4 - maxpool from 4 values, d - dense/fc layer
    optimizer: name (eg adam) followed by learning rate (with 0. removed, eg 0.001 -> 001
    patient: added to the end on save
    """

    # Model
    model = models.CNN1min(out1=c1, out2=c2, out3=fc1)
    # print(model)
    # print(list(model.parameters()))
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_path = "/media/projects/daniel_lstm/models/" + model_name + "_%d.pt" % pt

    # Train model
    if train:
        # model.to(device)
        train_model(model, gd.BalancedData1m(pt=pt, stepback=2, multiple=data_mult, duplicate_ictal=duplicate_ictal),
                    gd.BalancedData1m(pt=pt, train=False, stepback=2))

    # Load model
    trained_model = load_model(model_path)

    if test:
        auc_a = np.zeros(test_iterations)
        lo_a = np.zeros(test_iterations)
        hi_a = np.zeros(test_iterations)
        bs_a = np.zeros(test_iterations)
        bss_a = np.zeros(test_iterations)
        bss_se_a = np.zeros(test_iterations)

        for j in range(test_iterations):
            y, yhat = load_or_calculate_forecasts(pt, trained_model, gd.BalancedData1m(pt=pt, train=False))

            sz_yhat, inter_yhat = met.split_yhat(y, yhat)

            if show_auc:
                a, lo, hi = met.auc_hanleyci(sz_yhat, inter_yhat)

                # a_s, a_e= met.auc_on_shuffled(1000, sz_yhat, inter_yhat)
                # print('Shuffled auc %0.3f (%0.4f)' % (a_s, a_e))
                auc_a[j] = a
                lo_a[j] = lo
                hi_a[j] = hi

            if show_bss:
                bs, bs_ref, bss, bss_se = met.brier_skill_score(sz_yhat, inter_yhat, p_sz=0.5)
                # print('Brier: %.3g, Bss: %0.3g (%.3g) Ref: %.3g' % (bs, bss, bss_se, bs_ref))
                bs_a[j] = bs
                bss_a[j] = bss
                bss_se_a[j] = bss_se

        print('-------RESULTS-------')
        if show_auc:
            ind = np.where(auc_a == np.median(auc_a))[0][0]
            print(auc_a)
            print('AUC: %0.3f (%0.3f:%0.3f)' % (auc_a[ind], lo_a[ind], hi_a[ind]))
        if show_bss:
            ind = np.where(bss_a==np.median(bss_a))[0][0]
            print(bss_a)
            print('Brier: %.3g, Bss: %0.3g (%.3g)' % (bs_a[ind], bss_a[ind], bss_se_a[ind]))
