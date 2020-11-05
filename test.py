import numpy as np
import torch, torchaudio, torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import os
import visualize as vis
import get_datasets as gd
import scipy
from scipy import signal
import create_datasets as cd
import metrics as met
import models

# for pt in [6,8,9,10,11,13,15]:
#
#     f = h5py.File('/media/NVdata/SzTimes/temp/all_train_2step_80_%d.mat' % pt, 'r')
#     print(f)
#     labels = np.asarray(f['sample_labels'])
#
#
#     # # y_t = torch.load('/media/projects/daniel_lstm/forecasts_validation/y_min_%d.pt' % pt)
#     # yhat_t = torch.load('/media/projects/daniel_lstm/forecasts_validation/yhat_medium_%d.pt' % pt)
#     # # print(yhat)
#     # f = h5py.File('/media/NVdata/SzTimes/all_test_2step_80_%d.mat' % pt, 'r')
#     # y = np.asarray(f['sample_labels'])
#     # f.close()
#     #
#     # # y = np.zeros(y_10.size*10)
#     # # for i in range(y_10.size):
#     # #     y[i*10:(i+1)*10]=y_10[i]
#     #
#     # # y = y_t.detach().numpy()
#     # yhat = yhat_t.detach().numpy()
#     #
#     #
#     # # plt.plot(y)
#     # # plt.show()
#     #
#     # p_sz = len(y[y==1]) / (len(y[y==1]) + len(y[y==0]))
#     # # print('Psz: %.3f' % p_sz)
#     #
#     # # print('split yhat')
#     # sz_yhat, inter_yhat = met.split_yhat(y, yhat)
#     #
#     # # print('Calculating AUC')
#     # a, lo, hi = met.auc_hanleyci(sz_yhat, inter_yhat)
#     # # print('Calculating BSS')
#     # bs, bs_ref, bss, bss_se = met.brier_skill_score(sz_yhat, inter_yhat, p_sz=p_sz)
#     # # print('Calculating Sensitivity')
#     # sen, tiw, zero_chance = met.sens_tiw(y, yhat, extrapolate=True)
#     #
#     # print('Pt %d:  AUC: %.3f (%.3f, %.3f), Bss: %.3f (%.3f), Sens: %.3f, TiW: %.3f, 0risk: %.3f' % (pt, a, lo, hi, bss, bss_se, sen, tiw, zero_chance))
#
#
#
#
# # data = gd.BalancedData(1)


def fill_with_noise(pt, data):
    # print('In NOISE')
    stats = np.load('/media/projects/daniel_lstm/raw_eeg_stats/%d.npy' % pt)
    # print('Data shape', data.shape)
    k=0

    # ind = np.asarray(np.where(np.isnan(data)))

    # #
    for i in range(data.shape[0]):
        ind = np.where(np.isnan(data[i]))
        data[i, ind] = np.random.normal(stats[i,0], stats[i, 1], len(ind))

    return data
#
#
#
x = np.random.random((16, 23960))
for ch in range(16):
    ind = np.random.choice(x.shape[1], int(x.shape[1]*.4), replace=False)  # 40% nans
    x[ch][ind] = np.nan

t0 = time.time()
print(np.sum(np.isnan(x)))
x_f = fill_with_noise(1, x)
print(np.sum(np.isnan(x_f)))
t = time.time()-t0
print('%.3g milliseconds' % (t*1000))

print(np.random.normal())

x = np.array([1,2,3,4,5,6])
x[[1,4]] = [8,9]
print(x)
