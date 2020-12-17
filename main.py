#!/usr/bin/env python3
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import time
import sys, os
import types

# import matplotlib.pyplot as plt

import create_datasets as cd
import models
import get_datasets as gd
import metrics as met
import traintest as tt
import model_functions as mf

# -------------- Options -------------------
patients = [1, 6, 8, 9, 10, 11, 13, 15]  # patient list (1-15)
# patients = [13, 15]  # patient list (1-15)

# patients = [1,6, 8, 9]  # patient list (1-15)
# patients = [15]  # patient list (1-15)

runs = 16

# model_type = '1min'
model_type = 'short'
# model_type = 'medium'
# model_type = 'long'
# model_type = 'combo'

train = 1   # binary, whether to train a new model
test = 1
test_on_whole = 0
use_existing_results = 0  # determines whether existing results should be gathered
test_iterations = 3  # odd, How many test sets to take median from

show_auc = 1
show_bss = 1
show_tiw = 1

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------- Hyperparameters ---------------

# combo_version = 1
# long_version = 6 # Basic (accidentaly overwritten), last_layer_off = 1
# medium_version = 8  # Basic = 2, last_layer_off = 3
# short_version = 15# Baisc = 8, last_layer off = 10
# min_version = 12  # Latest: flatspec:11, spec:10, timeseries:10, untrained: 0

combo_version = 2
long_version = 10  # Basic (accidentaly overwritten), last_layer_off = 1
medium_version = 12  # Basic = 2, last_layer_off = 3
short_version = 21  # Baisc = 8, last_layer off = 10
min_version = 13  # Latest: flatspec:11, spec:10, timeseries:10, untrained: 0

sample_window = 10  # original was 10 minutes, options: 2min, 4min, 10min, 20min, 60min
data_mult = 1  # how many interictal sample per seizure (this is balanced by including duplicate sz samples)
duplicate_ictal = False
use_spec = True
transform = None
flatten_spec = False
nan_as_noise = True
short_look_back_min = 60
short_look_back = int(short_look_back_min / sample_window) # how many samples prior to labeled sample to start training from (eg 6
hrs_back = 24
days_back = 30

n_epochs = 5
min_batch_size = 16*sample_window  # for 1mBalanced: 20 samples / sz
batch_size = 16
learning_rate = .001

# 1 min

min_c1 = 16  # 15
min_c2 = 32  # 64
min_c3 = 64
min_fc1 = 32  # 128
min_fc2 = 16

# variables to iterate over
if runs>1:
    model_type_ = ['1min', 'short', 'medium', 'long',
                   '1min', 'short', 'medium', 'long',
                   '1min', 'short', 'medium', 'long',
                   '1min', 'short', 'medium', 'long']
    long_version_ = [11, 11, 11, 11,
                     12, 12, 12, 12,
                     13, 13, 13, 13,
                     14, 14, 14, 14]
    medium_version_ =[13, 13, 13, 13,
                      14, 14, 14, 14,
                      15, 15, 15, 15,
                      16, 16, 16, 16]
    short_version_ = [25, 25, 25, 25,
                      26, 26, 26, 26,
                      27, 27, 27, 27,
                      28, 28, 28, 28]
    min_version_ = [18, 18, 18, 18,
                    19, 19, 19, 19,
                    20, 20, 20, 20,
                    21, 21, 21, 21]
    sample_window_ = [2, 2, 2, 2,
                      4, 4, 4, 4,
                      10, 10, 10, 10,
                      20, 20, 20, 20]
    data_mult_ = [20, 20, 20, 20,
                  10, 10, 10, 10,
                  4,  4,  4,  4,
                  2,  2,  2,  2]
    n_epochs_ = [2, 5, 5, 5,
                2, 5, 5, 5,
                2, 5, 5, 5,
                2, 5, 5, 5]



def train_model(untrained_model, dataset, test_dataset, model_name, model_path, batch_size):
    # print('IN train model')
    batches_per_epoch = np.ceil(dataset.len / batch_size)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    out_model = tt.train(model=untrained_model, train_loader=train_loader, test_loader=test_loader, criterion=criterion,
                         optimizer=optimizer, n_epochs=n_epochs, batches_per_epoch=batches_per_epoch, batch_size=batch_size,
                         model_name=model_name + '_%d' % pt)
    print('Model saved as: ', model_path)
    torch.save(out_model, model_path)


def load_or_calculate_forecasts(pt, t_model, dataset, model_name):
    dir_path = '/media/projects/daniel_lstm/results_log/' + model_name
    file_path = dir_path + '/%d.pt' % pt
    results_not_found = False

    if use_existing_results and not train and os.path.isfile(file_path):  # Always want new results if retrained
        y, yhat = torch.load(file_path)


            # sys.exit(0)
    else:
        validation_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        batches_per_epoch = np.ceil(dataset.len/batch_size)
        print('Testing')
        y, yhat = tt.test(pt, t_model, validation_loader, batches_per_epoch=batches_per_epoch)
        print('\n')

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save([y, yhat], file_path)

    return y, yhat


for run in range(runs):
    model_type = model_type_[run]
    long_version = long_version_[run]
    medium_version = medium_version_[run]
    short_version = short_version_[run]
    min_version = min_version_[run]
    sample_window = sample_window_[run]
    data_mult = data_mult_[run]
    n_epochs = n_epochs_[run]
    min_batch_size = 16 * sample_window  # for 1mBalanced: 20 samples / sz

    print('\n\n######################## NEW RUN ########################')
    print('Type:', model_type)
    print('Window:', sample_window)
    print('Version:', short_version)
    if train:
        print('Training')
    if test:
        print('Testing')

    trans_text = ''
    if use_spec:
        if flatten_spec:
            transform = cd.flattened_spectrogram
            trans_text = '_fspec'
        else:
            transform = cd.spectrogram
            trans_text = '_spec'

    aucs = np.zeros(15)
    los = np.zeros(15)
    his = np.zeros(15)
    bsss = np.zeros(15)
    bsses = np.zeros(15)
    sens = np.zeros(15)
    tiws = np.zeros(15)
    zero_chances = np.zeros(15)

    for pt in patients:
        print('\n---------------- Patient %d ------------------' % pt)

        if pt == 11 or pt == 13:
            learning_rate = .001

        # ---------- Model Name/Path ----------
        min_model_name = '1m_v%d_c%dp4c%dp4d%dd_%d_adam' % (min_version, min_c1, min_c2, min_fc1, min_batch_size) + str(
            learning_rate)[2:] + trans_text
        short_model_name = 'short_v%d_' % (short_version)  # changing the naming convention as it will get too messy, just updating version every time
        medium_model_name = 'medium_v%d_' % (medium_version)
        long_model_name = 'long_v%d_' % (long_version)
        combo_model_name = 'combo_v%d_' % (combo_version)

        min_model_path = "/media/projects/daniel_lstm/models/" + min_model_name + "_%d.pt" % pt
        short_model_path = "/media/projects/daniel_lstm/models/" + short_model_name + "_%d.pt" % pt
        medium_model_path = "/media/projects/daniel_lstm/models/" + medium_model_name + "_%d.pt" % pt
        long_model_path = "/media/projects/daniel_lstm/models/" + long_model_name + "_%d.pt" % pt
        combo_model_path = "/media/projects/daniel_lstm/models/" + combo_model_name + "_%d.pt" % pt

        # ---------- Build Model ----------

        if model_type == '1min':

            if use_spec:
                if flatten_spec:
                    model = models.CNN1min_fspec(out1=min_c1, out2=min_c2, out3=min_c3, out4=min_fc1, out5=min_fc2)
                else:
                    model = models.CNN1min_spec(out1=min_c1, out2=min_c2, out3=min_c3, out4=min_fc1, out5=min_fc2)
            else:
                model = models.CNN1min(out1=min_c1, out2=min_c2, out3=min_fc1)

        elif model_type == 'short':
            model = models.Short(min_model_path=min_model_path, rn1=16, out1=16, transform=transform, lookBack=short_look_back, sample_window=sample_window)
        elif model_type == 'medium':
            model = models.Medium(min_model_path=min_model_path, rn1=16, out1=16, transform=transform, hrsBack=hrs_back, sample_window=sample_window)
        elif model_type == 'long':
            model = models.Long(min_model_path=min_model_path, rn1=16, out1=16, transform=transform, daysBack=days_back, sample_window=sample_window)
        elif model_type == 'combo':
            model = models.Combo(min_model_path, short_model_path, medium_model_path, long_model_path, transform=transform, sample_window=sample_window)
        else:
            print('Model Type not found')
            sys.exit(0)

        criterion = mf.weightedBCE([2. / (data_mult + 1.), 2. * data_mult / (data_mult + 1)])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # ---------- Train Model ----------

        if train:
            # if model_type==
            # model.to(device)
            # ----- HERE HERE HERE -----
            if model_type == '1min':
                train_model(model,
                            gd.BalancedSpreadData1m(pt=pt, sample_window= sample_window, stepback=2, multiple=data_mult, duplicate_ictal=duplicate_ictal,
                                                    transform=transform, nan_as_noise=nan_as_noise),
                            gd.BalancedSpreadData1m(pt=pt, sample_window= sample_window, train=False, stepback=2, transform=transform, nan_as_noise=nan_as_noise),
                            min_model_name, min_model_path, batch_size=min_batch_size)
            elif model_type == 'short':
                train_model(model,
                            dataset=gd.BalancedData(pt=pt, sample_window= sample_window, stepback=2, transform=transform, lookBack=short_look_back, nan_as_noise=nan_as_noise),
                            test_dataset=gd.BalancedData(pt=pt, sample_window= sample_window, stepback=2, transform=transform, train=False, lookBack=short_look_back, nan_as_noise=nan_as_noise),
                            model_name=short_model_name,
                            model_path=short_model_path,
                            batch_size=batch_size
                            )
                print('')
            elif model_type == 'medium':
                train_model(model,
                            dataset=gd.BalancedData(pt=pt, sample_window=sample_window, train=True, stepback=2, transform=transform, medium=True, nan_as_noise=nan_as_noise),
                            test_dataset=gd.BalancedData(pt=pt, sample_window=sample_window, train=False, stepback=2, transform=transform, medium=True, nan_as_noise=nan_as_noise),
                            model_name=medium_model_name,
                            model_path=medium_model_path,
                            batch_size=batch_size
                            )
            elif model_type == 'long':
                train_model(model,
                            dataset=gd.BalancedData(pt=pt, sample_window=sample_window, train=True, stepback=2, transform=transform, long=True, nan_as_noise=nan_as_noise),
                            test_dataset=gd.BalancedData(pt=pt, sample_window=sample_window, train=False, stepback=2, transform=transform, long=True, nan_as_noise=nan_as_noise),
                            model_name=long_model_name,
                            model_path=long_model_path,
                            batch_size=batch_size
                            )
            elif model_type == 'combo':
                train_model(model,
                            dataset = gd.BalancedDataCombo(pt=pt, sample_window=sample_window, train=True, transform=transform, lookBack=6, nan_as_noise=nan_as_noise),
                            test_dataset = gd.BalancedDataCombo(pt=pt, sample_window=sample_window, train=False, transform=transform, lookBack=6, nan_as_noise=nan_as_noise),
                            model_name=combo_model_name,
                            model_path=combo_model_path,
                            batch_size=batch_size
                            )
            else:
                print('Model type not found')
                sys.exit(0)


        # ---------- Test Model ----------

        if test:

            # Load model
            if model_type == '1min':
                trained_model = models.load_model(min_model_path)
            elif model_type == 'short':
                trained_model = models.load_model(short_model_path)
                # trained_model.forward = types.MethodType(tt.replacement_forward_short, trained_model)
            elif model_type == 'medium':
                trained_model = models.load_model(medium_model_path)
                # trained_model.forward = types.MethodType(tt.replacement_forward_medium, trained_model)
            elif model_type == 'long':
                trained_model = models.load_model(long_model_path)
                # trained_model.forward = types.MethodType(tt.replacement_forward_long, trained_model)
            elif model_type == 'combo':
                trained_model = models.load_model(combo_model_path)
            else:
                print('Model type not found')
                sys.exit(0)

            auc_a = np.zeros(test_iterations)
            lo_a = np.zeros(test_iterations)
            hi_a = np.zeros(test_iterations)
            bs_a = np.zeros(test_iterations)
            bss_a = np.zeros(test_iterations)
            bss_se_a = np.zeros(test_iterations)
            sen_a = np.zeros(test_iterations)
            tiw_a = np.zeros(test_iterations)
            zero_chance_a = np.zeros(test_iterations)

            for j in range(test_iterations):
                if model_type == '1min':
                    test_data = gd.BalancedSpreadData1m(pt=pt, sample_window=sample_window, train=False, transform=transform, nan_as_noise=nan_as_noise)
                    model_name = min_model_name
                elif model_type == 'short':
                    test_data = gd.BalancedData(pt=pt, sample_window=sample_window, train=False, transform=transform, lookBack=short_look_back, nan_as_noise=nan_as_noise)
                    model_name = short_model_name
                elif model_type == 'medium':
                    test_data = gd.BalancedData(pt=pt, sample_window=sample_window, train=False, transform=transform, medium=True, nan_as_noise=nan_as_noise)
                    model_name = medium_model_name
                elif model_type == 'long':
                    test_data = gd.BalancedData(pt=pt, sample_window=sample_window, train=False, transform=transform, long=True, nan_as_noise=nan_as_noise)
                    model_name = long_model_name
                elif model_type == 'combo':
                    test_data = gd.BalancedDataCombo(pt=pt, sample_window=sample_window, train=False, transform=transform, lookBack=6, nan_as_noise=nan_as_noise)
                    model_name = combo_model_name
                else:
                    print('Model not found during valedation', model_type)
                    sys.exit(0)
                y, yhat = load_or_calculate_forecasts(pt, trained_model, test_data, model_name)

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
                if show_tiw:
                    sen, tiw, zero_chance = met.sens_tiw(y, yhat, extrapolate=True)
                    sen_a[j] = sen
                    tiw_a[j] = tiw
                    zero_chance_a[j] = zero_chance

            print('-------RESULTS-------')
            if show_auc:
                ind = np.where(auc_a == np.median(auc_a))[0][0]
                # print(auc_a)
                aucs[pt - 1] = auc_a[ind]
                los[pt - 1] = lo_a[ind]
                his[pt - 1] = hi_a[ind]

                print('AUC: %0.3f (%0.3f:%0.3f)' % (auc_a[ind], lo_a[ind], hi_a[ind]))
            if show_bss:
                ind = np.where(bss_a == np.median(bss_a))[0][0]
                bsss[pt - 1] = bss_a[ind]
                bsses[pt - 1] = bss_se_a[ind]
                # print(bss_a)
                print('Brier: %.3g, Bss: %0.3g (%.3g)' % (bs_a[ind], bss_a[ind], bss_se_a[ind]))
            if show_tiw:
                ind = np.where(auc_a == np.median(auc_a))[0][0]
                sens[pt - 1] = sen_a[ind]
                tiws[pt - 1] = tiw_a[ind]
                zero_chances[pt - 1] = zero_chance_a[ind]
                print('Sens: %0.3f, TiW: %0.3f, zeroCh: %0.3f)' % (sen_a[ind], tiw_a[ind], zero_chance_a[ind]))

        if test_on_whole:

            # Load model
            if model_type == '1min':
                dataset = gd.WholeDataset(pt, train=False, transform=transform, nan_as_noise=nan_as_noise)
                data_loader = DataLoader(dataset, batch_size=1)
                min_model = models.load_model(min_model_path)
            elif model_type == 'short':
                dataset = gd.WholeDataset(pt, train=False, transform=None, nan_as_noise=nan_as_noise)
                data_loader = DataLoader(dataset, batch_size=1)
                short_model = models.load_model(short_model_path)
                short_model.lookBack = 1
                # trained_model.forward = types.MethodType(tt.replacement_forward_short, trained_model)
            elif model_type == 'medium':
                dataset = gd.WholeDataset(pt, train=False, transform=None, lastOnly=True, nan_as_noise=nan_as_noise)
                data_loader = DataLoader(dataset, batch_size=1)
                medium_model = models.load_model(medium_model_path)
                medium_model.hrsBack = 0

                # trained_model.forward = types.MethodType(tt.replacement_forward_medium, trained_model)
            elif model_type == 'long':
                dataset = gd.WholeDataset(pt, train=False, transform=None, lastOnly=True, nan_as_noise=nan_as_noise)
                data_loader = DataLoader(dataset, batch_size=1)
                long_model = models.load_model(long_model_path)
                long_model.days_back = 0
            elif model_type == 'combo':
                combo_model = models.load_model(combo_model_path)
            else:
                print('Model type not found')
                sys.exit(0)

            print(len(dataset), ' samples')


            if model_type == '1min' or model_type == 'short':
                y_tensor = torch.tensor(np.zeros(len(dataset)*10), dtype=torch.float)
                yhat_tensor = torch.tensor(np.zeros(len(dataset)*10), dtype=torch.float)
            else:
                y_tensor = torch.tensor(np.zeros(len(dataset)), dtype=torch.float)
                yhat_tensor = torch.tensor(np.zeros(len(dataset)), dtype=torch.float)
            i=0
            t_o = time.time()
            total_days = len(dataset)/144

            h_short = torch.randn(1, 1, 16)
            c_short = torch.randn(1, 1, 16)

            h_medium = [torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16)]
            c_medium = [torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16)]

            h_long = [torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        ]
            c_long = [torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 16),
                        ]


            # print('h_medium shape', h_medium)

            for i, (x, y) in enumerate(data_loader):

                if i>0:
                    t_d = time.time() - t_o
                    day = i/144.
                    percent_done = float(i) / len(dataset)
                    time_total = t_d / float(i) * len(dataset)
                    print('Pt %d. Sample %d of %d tested' % (pt, i, len(dataset)), file=open('/media/projects/daniel_lstm/tmp/log.txt', 'w'))
                    print('\rDay %.1f of %.1f. %.0f of %.0f seconds' % (day, total_days, t_d, time_total), end="")
                    if i%100 == 0:
                        # Save to file in case of crash
                        # Save model
                        # Save h and c (for LSTMs)
                        # Save yhat and y tensors
                        torch.save(y_tensor, '/media/projects/daniel_lstm/tmp/y_tmp.pt')
                        torch.save(yhat_tensor, '/media/projects/daniel_lstm/tmp/yhat_tmp.pt')
                        # Save progress to log file


                # x, y = x.to(device), y.to(device)

                if model_type == '1min':
                    name = '1min_v%d' % min_version
                    y_tensor[i * 10:(i + 1) * 10] = y.type(torch.float)
                    for min in range(10):
                        x_min = x[:,min]

                        # print(x_min.detach().numpy().shape)

                        yhat = min_model(x_min)

                        tmp = yhat_tensor.detach().numpy()
                        tmp[i*10+min] = yhat.detach().numpy()

                        yhat_tensor = torch.tensor(tmp, dtype=float)
                elif model_type == 'short':
                    name = 'short_v%d' % short_version

                    y_tensor[i * 10:(i + 1) * 10] = y.type(torch.float)

                    # print('Incoming data shape ', x.detach().numpy().shape)
                    # print('Incoming hidden shape', h_short.detach().numpy().shape)
                    print('h short ', h_short)
                    yhat, h_short, c_short = short_model(x, h0=h_short, c0=c_short, save_hidden=True)
                    tmp = yhat_tensor.detach().numpy()
                    tmp[i * 10: (i+1) * 10] = yhat.detach().numpy()
                    yhat_tensor = torch.tensor(tmp, dtype=float)

                elif model_type == 'medium':
                    name = 'med_v%d' % medium_version

                    y_tensor[i] = y.type(torch.float)

                    # min = x[:, -1]
                    j = i%6  # 6 parallel memories each takes one sample per hour
                    # print('h_med ',i, ' ', h_medium[j])

                    yhat, h_medium[j], c_medium[j] = medium_model(x, h0=h_medium[j], c0=c_medium[j], save_hidden=True)
                    # print('h_med ', i, ' ', h_medium[j])
                    tmp = yhat_tensor.detach().numpy()
                    tmp[i] = yhat.detach().numpy()
                    yhat_tensor = torch.tensor(tmp, dtype=float)

                elif model_type == 'long':
                    name = 'long_v%d' % long_version

                    y_tensor[i] = y.type(torch.float)
                    if i%6==0:  # only take one an hour
                        # min = x[:, -1]
                        j = int(i/6%24)  # 24 parallel memories each takes one sample per hour
                        # print('i: %d, j: %d' % (i, j))
                        # print('h_long ', i, ' ', h_long[j])

                        yhat, h_long[j], c_long[j] = long_model(x, h0=h_long[j], c0=c_long[j], save_hidden=True)
                        # print('h_long ', i, ' ', h_long[j])
                        tmp = yhat_tensor.detach().numpy()
                        tmp[i] = yhat.detach().numpy()
                        yhat_tensor = torch.tensor(tmp, dtype=float)
                    else:
                        tmp = yhat_tensor.detach().numpy()
                        tmp[i] = tmp[i-1]
                        yhat_tensor = torch.tensor(tmp, dtype=float)


            torch.save(y_tensor, '/media/projects/daniel_lstm/forecasts_validation/y_' + name + '_%d.pt' % pt)
            torch.save(yhat_tensor, '/media/projects/daniel_lstm/forecasts_validation/yhat_' + name + '_%d.pt' % pt)



    print('\n\n ------------ RESULTS --------------')
    print('Type:', model_type)
    print('Window:', sample_window)
    print('Version:', short_version)
    for pt in patients:
        print('Patient %d, AUC: %.3f (%.3f, %.3f), BSS: %.3g (%.3g), Sens: %.3f, TiW: %.3f, 0risk: %.3f' %
              (pt, aucs[pt - 1], los[pt - 1], his[pt - 1], bsss[pt - 1], bsses[pt - 1], sens[pt - 1], tiws[pt - 1],
               zero_chances[pt - 1]))

