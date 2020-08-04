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
import model_functions as mf

# -------------- Options -------------------
patients = [1, 6, 8, 9, 10, 11, 13, 15]  # patient list (1-15)
# patients = [6, 8, 9, 10, 11, 13, 15]  # patient list (1-15)

# patients = [6, 9, 10, 11]  # patient list (1-15)

# patients = [6]  # patient list (1-15)

model_type = '1min'
model_type = 'short'
# model_type = 'medium'
# model_type = 'long'

train = 0                              # binary, whether to train a new model
test = 1                            # binary, whether to test the model
use_existing_results = 0            # determines whether existing results should be gathered
test_iterations = 3                 # odd, How many test sets to take median from

show_auc = 1
show_bss = 1
show_tiw = 1

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------- Hyperparameters ---------------

version = 8
min_version= 10 # Latest: flatspec:11, spec:10, timeseries:10

data_mult = 1  # how many interictal sample per seizure (this is balanced by including duplicate sz samples)
duplicate_ictal = False
use_spec = True
transform = None
flatten_spec = False
look_back = 6  # how many samples prior to labeled sample to start training from (eg 6

n_epochs = 3
min_batch_size = 160  # for 1mBalanced: 20 samples / sz
batch_size = 16
learning_rate = .001

# 1 min

min_c1=16  #15
min_c2=32  #64
min_c3 =64
min_fc1=32  #128
min_fc2 = 16

# short




def train_model(untrained_model, dataset, test_dataset, model_name, model_path):
    batches_per_epoch = np.ceil(dataset.len / batch_size)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_dataset.len)
    out_model = tt.train(model=untrained_model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer,
                                 n_epochs=n_epochs, batches_per_epoch=batches_per_epoch, model_name=model_name + '_%d' % pt)
    print('Model saved as: ', model_path)
    torch.save(out_model, model_path)

def load_or_calculate_forecasts(pt, t_model, dataset, model_name):
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

trans_text = ''
if use_spec:
    if flatten_spec:
        transform = cd.flattened_spectrogram
        trans_text = '_fspec'
    else:
        transform=cd.spectrogram
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

    if pt==11 or pt==13:
        learning_rate = .001


    # ---------- Model Name/Path ----------
    # ----- HERE HERE HERE -----
    min_model_name = '1m_v%d_c%dp4c%dp4d%dd_%d_adam' % (min_version, min_c1, min_c2, min_fc1, min_batch_size) + str(learning_rate)[2:] + trans_text
    short_model_name = 'short_v%d_' % (version) # changing the naming convention as it will get too messy, just updating version every time


    min_model_path = "/media/projects/daniel_lstm/models/" + min_model_name + "_%d.pt" % pt
    short_model_path = "/media/projects/daniel_lstm/models/" + short_model_name + "_%d.pt" % pt

    # ---------- Build Model ----------

    if model_type=='1min':
        if use_spec:
            if flatten_spec:
                model = models.CNN1min_fspec(out1=min_c1, out2=min_c2, out3=min_c3, out4=min_fc1, out5=min_fc2)
            else:
                model = models.CNN1min_spec(out1=min_c1, out2=min_c2, out3=min_c3, out4=min_fc1, out5=min_fc2)
        else:
            model = models.CNN1min(out1=min_c1, out2=min_c2, out3=min_fc1)
    elif model_type=='short':
        model = models.Short(min_model_path=min_model_path, rn1=16, out1=16, transform=transform, lookBack=look_back)
    else:
        print('Model Type not found')
        sys.exit(0)

    criterion = mf.weightedBCE([2./(data_mult+1.), 2.*data_mult/(data_mult+1)])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # ---------- Train Model ----------

    if train:
        # if model_type==
        # model.to(device)
        # ----- HERE HERE HERE -----
        if model_type=='1min':
            train_model(model, gd.BalancedSpreadData1m(pt=pt, stepback=2, multiple=data_mult, duplicate_ictal=duplicate_ictal, transform=transform),
                        gd.BalancedSpreadData1m(pt=pt, train=False, stepback=2, transform=transform), min_model_name, min_model_path)
        elif model_type=='short':
            train_model(model, gd.BalancedData(pt=pt, stepback=2, transform=transform, lookBack=look_back),
                        gd.BalancedData(pt=pt, stepback=2, transform=transform, train=False, lookBack=look_back), short_model_name, short_model_path)
            print('')
        else:
            print('Model type not found')
            sys.exit(0)

    # Load model
    if model_type=='1min':
        trained_model = models.load_model(min_model_path)
    elif model_type=='short':
        trained_model = models.load_model(short_model_path)
    else:
        print('Model type not found')
        sys.exit(0)


    # ---------- Test Model ----------

    if test:
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
            # ----- HERE HERE HERE -----
            if model_type=='1min':
                test_data = gd.BalancedSpreadData1m(pt=pt, train=False, transform=transform)
                model_name = min_model_name
            elif model_type=='short':
                test_data = gd.BalancedData(pt=pt, train=False, transform=transform, lookBack=look_back)
                model_name = short_model_name
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
            ind = np.where(bss_a==np.median(bss_a))[0][0]
            bsss[pt-1] = bss_a[ind]
            bsses[pt-1] = bss_se_a[ind]
            # print(bss_a)
            print('Brier: %.3g, Bss: %0.3g (%.3g)' % (bs_a[ind], bss_a[ind], bss_se_a[ind]))
        if show_tiw:
            ind = np.where(auc_a == np.median(auc_a))[0][0]
            sens[pt-1] = sen_a[ind]
            tiws[pt-1] = tiw_a[ind]
            zero_chances[pt-1] = zero_chance_a[ind]
            print('Sens: %0.3f, TiW: %0.3f, zeroCh: %0.3f)' % (sen_a[ind], tiw_a[ind], zero_chance_a[ind]))


print('\n\n ------------  FINAL RESULTS --------------')
for pt in patients:
    print('Patient %d, AUC: %.3f (%.3f, %.3f), BSS: %.3g (%.3g), Sens: %.3f, TiW: %.3f, 0risk: %.3f' %
          (pt, aucs[pt-1], los[pt-1], his[pt-1], bsss[pt-1], bsses[pt-1], sens[pt-1], tiws[pt-1], zero_chances[pt-1]))