import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import h5py

import get_datasets as gd
import create_datasets as cd


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


# # pt =  8
# # jump = 10
# # start_s=0 + jump*1000
# # end_s=10 + jump*1000
# #
# # start = 8640000 + start_s * 600
# # end = 8640000+ end_s * 600
# # data = cd.get_data(pt, start, end)
# # raw_eeg(pt, data, start, end)

for p in [1,6,8,9,10,11,13,15]:
    print(p)
    sz_types(p)