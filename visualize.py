import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import h5py

import preprocessing as pre



def raw_eeg(pt, eeg_data, t_start, t_end, sznum='', annots={}):

    rec_start = pre.get_record_start(pt)

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

    fs = pre.get_fs(pt)
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

    raw = pre.get_data(patient, t_start, t_end)

    raw_eeg(patient, raw, t_start, t_end, sznum, {'sz':szTime})