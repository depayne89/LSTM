import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats as stats
import torch

import datetime
import time
import h5py

import get_datasets as gd
import create_datasets as cd

rc('text', usetex=False)

def raw_eeg(pt, eeg_data, t_start, t_end, sznum='', annots={}):

    rec_start = cd.get_record_start(pt)

    start = rec_start + t_start
    end = rec_start + t_end


    height = 200
    length = eeg_data.shape[1]
    t = np.linspace(start, end, num=length)
    print(length, t.shape)


    n_ticks = 5
    labels = []
    ticks = np.zeros(n_ticks+1)
    j=0
    for i, sec in enumerate(t):
        if i%np.floor(length/n_ticks)==0:
            dt = time.localtime(t[i])
            label = '%d:%d:%d' % (dt.tm_hour, dt.tm_min, dt.tm_sec)
            labels.append(label)
            ticks[j] = i
            j+=1
            print(i, j, label, t[i])

    print(labels)
    for ch in range(16):
        print(ch)
        to_plot = eeg_data[ch, :] + height + 2 * height * ch
        plt.plot(to_plot, 'k', linewidth=0.5)

    fs = cd.get_fs(pt)
    for annot, secs in annots.items():
        sec_in_fig = secs-t_start
        step = round(sec_in_fig*fs)  # FIX TO GET CORRECT FS?
        print('Step: ', step)
        plt.axvline(x=step)


    start_time = time.localtime(t[0])
    date = '%d/%d/%d' % (start_time.tm_mday, start_time.tm_mon, start_time.tm_year)

    plt.title('Pt %d ' %pt + date)
    plt.xticks(ticks, labels)
    plt.savefig('./figs/rawEEG_' + str(pt) + '_' + str(sznum))


def raw_sz_eeg(patient, sznum, t_before, t_after, inter=False, inter_jump = 60*60*2, train=True):

    # Load sz times
    f = h5py.File('/media/NVdata/SzTimes/50_50_%d.mat' % patient, 'r')

    if train:
        szTimes = np.asarray(f['train'].value)
    else:
        szTimes = np.asarray(f['test'].value)

    szTime = szTimes[sznum]
    t_start = szTime - t_before
    t_end = szTime + t_after
    if inter:
        t_start = t_start - inter_jump
        t_end = t_end - inter_jump

    raw = gd.get_data(patient, t_start, t_end)

    raw_eeg(patient, raw, t_start, t_end, sznum, {'sz':szTime})


def moving_average(x, n=5):

    out = np.zeros(x.shape[0]-n+1)

    for i in range(out.shape[0]):
        out[i] = np.mean(x[i:i+n])

    return out


def see_drop(patient, train=True, percent_train=80, steps_back=2):


    if train:
        f = h5py.File('/media/NVdata/SzTimes/all_train_%dstep_%d_%d.mat' % (steps_back, percent_train, patient), 'r')
        d_set = 'train'
    else:
        f = h5py.File('/media/NVdata/SzTimes/all_test_%dstep_%d_%d.mat' % (steps_back, percent_train, patient), 'r')
        d_set = 'test'

    dropouts = np.asarray(f['sample_dropout'])
    dropouts_smooth = moving_average(dropouts, 144)
    average = np.mean(dropouts)
    print('mean:', average)

    f.close()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dropouts_smooth)
    ax.axhline(average)
    # plt.show()

    plt.savefig('/home/daniel/Desktop/LSTM_figs/drop_in_' + d_set + '_%d.png' % patient, format='png')


def sz_types(pt):
    [SzDur, SzInd, SzTimes, SzType] = cd.get_annots(pt)

    print(SzTimes[0]/1000000./3600./24.)

    ISI = np.squeeze(SzTimes[1:] - SzTimes[:-1])
    print(ISI.shape)
    ISI = np.concatenate((np.array([np.inf]), ISI))

    ISI = ISI/1000000/60/60 # convert to hours
    leads = np.where(ISI>5)[0]
    burst = np.where(ISI<=5)[0]
    SzType[burst] = SzType[burst] - 0.2

    plt.plot(SzTimes[burst], SzType[burst], 'o')
    plt.plot(SzTimes[leads], SzType[leads], 'ro')
    plt.ylim([-5,10])
    plt.show()


def loss_and_auc(loss, auc, vloss, vauc, model_name, batches_per_epoch, epochs):
    fig, ax = plt.subplots(figsize=(8, 3))
    # ax.set_title('A', fontsize=9, loc='left', position = (-.25,1.05))
    ax.set_axis_off()

    val_ind = (np.arange(epochs) + 1) * batches_per_epoch - 1

    # auc
    auc_ax = fig.add_axes([.08, .15, .4, .8])
    auc_ax.set_ylabel('AUC')
    auc_ax.plot(auc, 'o', color='gray', markersize=4)
    auc_ax.plot(val_ind, vauc[val_ind.astype(int)], 'ro', markersize=5)
    auc_ax.set_ylim(bottom=0)
    auc_ax.legend(['Train', 'Test'], bbox_to_anchor=(.98,.25))
    auc_ax.set_xticks(val_ind)
    labels = np.array([])
    for i in np.arange(epochs):
        labels = np.append(labels, str(i+1))
    print(labels)
    auc_ax.set_xticklabels(labels)
    auc_ax.set_xlabel('Epochs')


    loss_ax = fig.add_axes([.58, .15, .4, .8])
    loss_ax.set_ylabel('Loss')
    loss_ax.plot(loss, color='gray')
    loss_ax.plot(val_ind, vloss[val_ind.astype(int)], 'ro', markersize=5)
    loss_ax.set_ylim(bottom=0)
    loss_ax.set_xticks(val_ind)
    loss_ax.set_xticklabels(labels)
    loss_ax.set_xlabel('Epochs')



    plt.savefig('/home/daniel/Desktop/LSTM_figs/losslogs/' + model_name)


def plot_sample(pt, data, label, prediction, sample_num):

    figpath = '/home/daniel/Desktop/LSTM_figs/eegs/%d/%d.png' % (pt, sample_num)

    if sample_num%10==0:
        print('%d samples proessed' % sample_num)

    fig, ax = plt.subplots()

    height = 200
    for ch in range(16):
        # print(ch)
        to_plot = data[ch, :] + height + 2 * height * ch
        ax.plot(to_plot, 'k', linewidth=0.5)

    label_text= 'Inter'
    if label==1:
        label_text = 'Seizure'
    correct = round(1 - abs(label - prediction))
    color='red'
    if correct:
        color = 'green'
    ax.text(.3, 1.02, label_text, transform=ax.transAxes, fontsize=18)
    ax.text(.55, 1.02, str(prediction), transform=ax.transAxes, color=color, fontsize=18)
    # plt.savefig(figpath)
    plt.show()



    test_title = 'Pt 5'

    dummy_data = np.array([1, 2, 3, 4])

    ax.plot(dummy_data)


def correlation(x, y, xlabel, ylabel):

    a, b, r, p, ste = stats.linregress(x, y)

    line_x = np.linspace(0,x.max(), 10)
    line_y = line_x * a + b

    fig, ax = plt.subplots()
    ax.plot(line_x, line_y, 'k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(x,y, 'bo')
    ax.set_title('R: %.2f, p:%.2g' %(r, p))
    plt.show()


def forecasts_plot(pt, yhat, y):
    """ Plots timeseries of every forecast over time


    :param yhat:
    :param y:
    :return:
    """

    sz_inds = np.where(y==1)[0]
    print(sz_inds)
    plt.plot(yhat, 'grey')
    plt.plot(sz_inds, np.ones(sz_inds.size), 'rv')
    plt.title(pt)


    plt.show()

def forecast_metrics(pt, name):


    y_t = torch.load('/media/projects/daniel_lstm/forecasts_validation/y_' + name + '_%d.pt' % pt)
    yhat_t = torch.load('/media/projects/daniel_lstm/forecasts_validation/yhat_' + name + '_%d.pt' % pt)

    y = y_t.detach().numpy()
    yhat = yhat_t.detach().numpy()

    # print(np.sum(y[y==1]))

    import metrics as met

    print('Splitting yhat')
    sz_yhat, inter_yhat = met.split_yhat(y, yhat)

    print('Calculating AUC')
    a, lo, hi = met.auc_hanleyci(sz_yhat, inter_yhat)

    print('Calculating BSS')
    bs, bs_ref, bss, bss_se = met.brier_skill_score(sz_yhat, inter_yhat, p_sz=0.5)

    sen, tiw, zero_chance = met.sens_tiw(y, yhat, extrapolate=False)
    print('Pt: %d. AUC: %0.3f (%0.3f:%0.3f), Brier: %.3g, Bss: %0.3g (%.3g), Sens: %0.3f, TiW: %0.3f, zeroCh: %0.3f)' % (pt, a, lo, hi, bs, bss, bss_se, sen, tiw, zero_chance))

    forecasts_plot(pt, yhat, y)

# forecast_metrics(1, '1min')
# for pt in [6, 8, 9]:
#     forecast_metrics(pt, '1min_v12')