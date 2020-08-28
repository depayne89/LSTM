import numpy as np
import time
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
import math

import create_datasets as cd


# ------------- Dataset Parameters -------------------

distance_from_sz = 2  # hours, minimum time between inter sample and sz
distance_from_inter = .5 # hours, minimum time between inter samples


# bandstop_at50

def select_interictal(self, multiple=1):
    num_sz = int(np.sum(self.labels[self.labels == 1])) * multiple
    length = self.labels.size
    sz_times = self.times[self.labels == 1]
    inter_times = np.zeros(num_sz)

    for i in range(num_sz):
        found = False
        # select random index and see if its far enough away
        while not found:
            k = np.random.randint(0, length)
            if self.labels[k] == 0:
                time_to_szs = np.abs(sz_times - self.times[k])
                if i > 0:
                    time_to_inter = np.abs(inter_times - self.times[k])
                    if time_to_szs.min() / 3600 > distance_from_sz and time_to_inter.min() / 3600 > distance_from_inter:
                        found = True
                else:
                    if time_to_szs.min() / 3600 > distance_from_sz:
                        found = True

        inter_times[i] = self.times[k]

    return inter_times


def shuffle_samples(sz_times, inter_times):
    times = np.concatenate((sz_times, inter_times))
    labels = np.concatenate((np.ones(sz_times.shape), np.zeros(inter_times.shape)))
    inds = np.arange(times.shape[0])
    np.random.shuffle(inds)

    return times[inds], labels[inds]


def zipper_samples(sz_times, inter_times):

    x = np.zeros(sz_times.size * 2)
    y = np.zeros(sz_times.size * 2)

    for i in range(sz_times.size):
        x[2 * i] = sz_times[i]
        y[2 * i] = 1
        x[2 * i + 1] = inter_times[i]

    return x, y

# def medium_times()

class WholeDataset_1min(Dataset):

    def __init__(self, iPt, train=True, train_percent=80, stepback=2,  transform=None):
        self.iPt = iPt
        self.transform = transform
        self.train = train
        self.train_percent = train_percent
        self.stepback=stepback
        self.rawDir = '/media/NVdata/Patient_' + cd.get_patient(iPt) + '/'
        self.set = 'train'
        if not train:
            self.set = 'test'

        f = h5py.File('/media/NVdata/SzTimes/all_' + self.set + '_2step_%d_%d.mat' % (train_percent, iPt), 'r')
        self.labels = np.asarray(f['sample_labels'])
        self.times = np.asarray(f['sample_times'])
        f.close()

    def __len__(self):
        return self.labels.size

    def test(self):
        print(self.labels)

    # def __len__(self):
    #     return self.len

    def __getitem__(self, idx):  # DATA doesn't have to be loaded until here!

        y = self.labels[idx]
        if y!=1:
            y=0

        start = self.times[idx]
        x = torch.tensor(np.zeros((10, 16, 120, 120)), dtype=torch.float)

        for i in range(10):
            end = start + 60

            tmp = cd.get_data(self.iPt, start, end)
            tmp_trim = tmp[:, :23960]
            tmp_trim[np.isnan(tmp_trim)] = 0

            tmp_tensor = torch.tensor(tmp_trim, dtype=torch.float)
            tmp_trans = self.transform(tmp_tensor)
            x[i] = tmp_trans
            start=end

        return x, torch.tensor(y, dtype=torch.float)



class WholeDataset(Dataset):

    def __init__(self, iPt, train=True, train_percent=80, stepback=2,  transform=None, lastOnly = False):
        self.iPt = iPt
        self.transform = transform
        self.train = train
        self.train_percent = train_percent
        self.stepback=stepback
        self.rawDir = '/media/NVdata/Patient_' + cd.get_patient(iPt) + '/'
        self.set = 'train'
        if not train:
            self.set = 'test'
        self.lastOnly = lastOnly

        f = h5py.File('/media/NVdata/SzTimes/all_' + self.set + '_2step_%d_%d.mat' % (train_percent, iPt), 'r')
        self.labels = np.asarray(f['sample_labels'])
        self.times = np.asarray(f['sample_times'])
        f.close()

    def __len__(self):
        return self.labels.size

    def test(self):
        print(self.labels)

    # def __len__(self):
    #     return self.len

    def __getitem__(self, idx):  # DATA doesn't have to be loaded until here!

        y = self.labels[idx]
        if y!=1:
            y=0

        start = self.times[idx]
        if self.transform is not None:
            x = torch.tensor(np.zeros((10, 16, 120, 120)), dtype=torch.float)
        else:
            x = torch.tensor(np.zeros((10,16,23960)), dtype=torch.float)

        if self.lastOnly:
            start += 540
            end = start + 60

            tmp = cd.get_data(self.iPt, start, end)
            tmp_trim = tmp[:, :23960]
            tmp_trim[np.isnan(tmp_trim)] = 0

            tmp_tensor = torch.tensor(tmp_trim, dtype=torch.float)
            if self.transform is not None:
                tmp_tensor = self.transform(tmp_tensor)
            x = tmp_tensor
        else:
            for i in range(10):
                end = start + 60

                tmp = cd.get_data(self.iPt, start, end)
                tmp_trim = tmp[:, :23960]
                tmp_trim[np.isnan(tmp_trim)] = 0

                tmp_tensor = torch.tensor(tmp_trim, dtype=torch.float)
                if self.transform is not None:
                    tmp_tensor = self.transform(tmp_tensor)
                x[i] = tmp_tensor
                start=end

        return x, torch.tensor(y, dtype=torch.float)



class BalancedData(Dataset):

    def __init__(self, pt, train=True, train_percent=80, stepback=2, transform=None, multiple=1, duplicate_ictal=False, lookBack=1, medium=False, long=False):
        self.pt=pt
        self.train_percent = train_percent
        self.stepback = stepback
        self.transform = transform
        self.train = train
        self.lookBack = lookBack
        self.medium = medium
        self.long = long
        if train:
            f = h5py.File('/media/NVdata/SzTimes/all_train_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
            self.start = f['train_start']
            self.end = f['train_end']
        else:
            f = h5py.File('/media/NVdata/SzTimes/all_test_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
            self.start = f['test_start']
            self.end = f['test_end']
        self.horizon = f['pred_horizon']
        self.times = np.asarray(f['sample_times'])
        self.labels = np.asarray(f['sample_labels'])
        self.dropout = np.asarray(f['sample_dropout'])
        f.close()

        # print('Selecting interictal samples')
        inter_times = select_interictal(self, multiple=multiple)
        # print('Complete')
        sz_times_base = self.times[self.labels==1]
        sz_times = sz_times_base
        if duplicate_ictal:
            for i in range(multiple - 1):
                sz_times = np.concatenate((sz_times, sz_times_base))

        self.select_times, self.select_labels = shuffle_samples(sz_times, inter_times)
        # self.zipper_samples(sz_times, inter_times)
        self.len = sz_times.size + inter_times.size

    def __len__(self):
        return self.len


    def __getitem__(self, idx):  # DATA doesn't have to be loaded until here!

        y = self.select_labels[idx]
        if y == 2 or y == -1:
            y = 0

        start = self.select_times[idx]
        end = start + 600

        if self.medium:
            # start -= 60*60 # REMOVE
            start += 60*9  # take last minute of the 10 min sample
            hrs_back = 24  # this means 25 samples as we go as far as exactly 24 hrs ago

            x_np = np.zeros((hrs_back+1, 16, 23960))
            
            for i in range(hrs_back+1):
                hrs = hrs_back-i
                min_start = start - hrs*60*60
                min_end = min_start + 60
                minute = cd.get_data(self.pt, min_start, min_end)
                x_np[i]=minute[:, :23960]
                

        elif self.long:
            # start -= 60*60*24
            start += 60 * 9
            days_back = 30  # days to look back

            x_np = np.zeros((days_back+1, 16, 23960))

            for i in range(days_back+1):
                days = days_back-i
                min_start = start - days*24*60*60
                min_end = min_start + 60
                minute = cd.get_data(self.pt, min_start, min_end)
                x_np[i] = minute[:, :23960]

        else:
            # start -= 60
            start -= 600 * (self.lookBack - 1)  # Go back 10s of minutes for more training info
            x_np = cd.get_data(self.pt, start, end)


        x_np[np.isnan(x_np)]=0
        # plt.plot(x_np[0])
        # plt.show()
        x = torch.tensor(x_np, dtype=torch.float32)
        # print('X size', x.size())


        return x, torch.tensor(y, dtype=torch.float32)


        # Randomly select interictal (class=0)
        # Check against requirements
            # 5 hrs from sz
            # Max dropout (0.5)
            # Min dropout (average / 4)
            # 2 hrs from any other interictal
        # Do so until matched to sz number


class BalancedData1m(Dataset):

    def __init__(self, pt, train=True, train_percent=80, stepback=2, transform=None, multiple=1, duplicate_ictal=False):
        self.pt=pt
        self.train=train
        self.train_percent=train_percent
        self.stepback=stepback
        self.transform=transform
        self.balancedData = BalancedData(pt, train, train_percent, stepback, transform, multiple, duplicate_ictal)
        self.original_length = self.balancedData.len
        self.len = self.original_length*10

    def __len__(self):
        return self.len

    def __getitem__(self, i):

        j = math.floor(i/10)
        k = i%10
        sample = self.balancedData[j]  # shape(2, 16, 239770) - x/y, channels, values
        # print('Sample shape', sample[0].size())
        y = sample[1] # labels

        one_min_lengths = [23977, 0, 0, 0, 0, 23967, 0, 23967, 23977, 23967, 23967, 0, 23977, 0, 23967]

        x = sample[0][:, 23960*k:23960*(k+1)]
        x_f = torch.tensor(cd.filter_data(x), dtype=torch.float32)

        if self.transform:
            # print(x.size)
            x = self.transform(x)
            # print('Spec shape', x.shape)
        # print('x', x[0, :10], x.type())
        # print('xf', x_f[0, :10], x_f.type())

        return x, y


class BalancedSpreadData1m(Dataset):

    def __init__(self, pt, train=True, train_percent=80, stepback=2, transform=None, multiple=1, duplicate_ictal = False):
        self.pt=pt
        self.train_percent = train_percent
        self.stepback = stepback
        self.transform = transform
        self.train = train
        if train:
            f = h5py.File('/media/NVdata/SzTimes/all_train_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
            self.start = f['train_start']
            self.end = f['train_end']
        else:
            f = h5py.File('/media/NVdata/SzTimes/all_test_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
            self.start = f['test_start']
            self.end = f['test_end']
        self.horizon = f['pred_horizon']
        self.times = np.asarray(f['sample_times'])
        self.labels = np.asarray(f['sample_labels'])
        self.dropout = np.asarray(f['sample_dropout'])
        f.close()

        sz_times = self.get_ictal()
        inter_times = self.get_inter()
        self.shuffle_samples(sz_times, inter_times)
        self.len = sz_times.size + inter_times.size

    def __len__(self):
        return self.len

    def get_ictal(self):
        sz_times = self.times[self.labels == 1]
        sz_times_split = np.zeros(sz_times.size*10)
        for i, time in enumerate(sz_times):
            for j in range(10):
                sz_times_split[i*10+j] = sz_times[i] + 60*j
        return sz_times_split

    def get_inter(self):
        num_sz = int(np.sum(self.labels[self.labels == 1]))
        length = self.labels.size
        sz_times = self.times[self.labels == 1]
        inter_times = np.zeros(num_sz*10)

        for i in range(num_sz*10):
            found = False
            # select random index and see if its far enough away
            while not found:
                k = np.random.randint(0, length)
                if self.labels[k] == 0:
                    time_to_szs = np.abs(sz_times - self.times[k])
                    if i > 0:
                        time_to_inter = np.abs(inter_times - self.times[k])
                        if time_to_szs.min() / 3600 > distance_from_sz and time_to_inter.min() / 3600 > distance_from_inter:
                            found = True
                    else:
                        if time_to_szs.min() / 3600 > distance_from_sz:
                            found = True

            inter_times[i] = self.times[k] + np.random.randint(0,9)*60  # choose random minute within the 10minute period

        return inter_times

    def shuffle_samples(self, sz_times, inter_times):

        times = np.concatenate((sz_times, inter_times))
        labels = np.concatenate((np.ones(sz_times.shape), np.zeros(inter_times.shape)))
        inds = np.arange(times.shape[0])
        np.random.shuffle(inds)

        self.select_times = times[inds]
        self.select_labels = labels[inds]

    def __getitem__(self, i):
        y = self.select_labels[i]
        if y == 2 or y == -1:
            y = 0

        start = self.select_times[i]
        end = start + 60
        x_np = cd.get_data(self.pt, start, end)
        x_np[np.isnan(x_np)] = 0
        # plt.plot(x_np[0])
        # plt.show()

        x_np = x_np[:, :23960]

        x = torch.tensor(x_np, dtype=torch.float32)

        if self.transform:
            # print(x.size)
            x = self.transform(x)
        # print('X size', x.size())

        return x, torch.tensor(y, dtype=torch.float32)


class BalancedDataCombo(Dataset):

    def __init__(self, pt, train=True, train_percent=80, stepback=2, transform=None, multiple=1, duplicate_ictal=False, lookBack=1, hrsBack=24, daysBack=30):
        self.pt=pt
        self.train_percent = train_percent
        self.stepback = stepback
        self.transform = transform
        self.train = train
        self.lookBack = lookBack
        self.hrsBack = hrsBack
        self.daysBack = daysBack
        if train:
            f = h5py.File('/media/NVdata/SzTimes/all_train_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
            self.start = f['train_start']
            self.end = f['train_end']
        else:
            f = h5py.File('/media/NVdata/SzTimes/all_test_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
            self.start = f['test_start']
            self.end = f['test_end']
        self.horizon = f['pred_horizon']
        self.times = np.asarray(f['sample_times'])
        self.labels = np.asarray(f['sample_labels'])
        self.dropout = np.asarray(f['sample_dropout'])
        f.close()

        # print('Selecting interictal samples')
        inter_times = select_interictal(self, multiple=multiple)
        # print('Complete')
        sz_times_base = self.times[self.labels==1]
        sz_times = sz_times_base
        if duplicate_ictal:
            for i in range(multiple - 1):
                sz_times = np.concatenate((sz_times, sz_times_base))

        self.select_times, self.select_labels = shuffle_samples(sz_times, inter_times)
        # self.zipper_samples(sz_times, inter_times)
        self.len = sz_times.size + inter_times.size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        y_np = self.select_labels[idx]
        if y_np == 2 or y_np == -1:
            y_np = 0

        start = self.select_times[idx]

        x_np = {}

        short_start = start - 600 * (self.lookBack - 1)  # Go back 10s of minutes for more training info

        x_np_short = cd.get_data(self.pt, short_start, start+600)
        x_np_short[np.isnan(x_np_short)] = 0
        x_np['short'] = x_np_short

        med_start = start + 60 * 9  # take last minute of the 10 min sample
        x_np_med = np.zeros((self.hrsBack + 1, 16, 23960))
        for i in range(self.hrsBack + 1):
            hrs = self.hrsBack - i
            min_start = med_start - hrs * 60 * 60
            min_end = min_start + 60
            minute = cd.get_data(self.pt, min_start, min_end)
            x_np_med[i] = minute[:, :23960]
        x_np_med[np.isnan(x_np_med)] = 0
        x_np['medium'] = x_np_med

        long_start = start + 60 * 9
        x_np_long = np.zeros((self.daysBack + 1, 16, 23960))
        for i in range(self.daysBack + 1):
            days = self.daysBack - i
            min_start = long_start - days * 24 * 60 * 60
            min_end = min_start + 60
            minute = cd.get_data(self.pt, min_start, min_end)
            x_np_long[i] = minute[:, :23960]
        x_np_long[np.isnan(x_np_long)] = 0
        x_np['long'] = x_np_long

        return x_np, y_np


