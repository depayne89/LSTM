import numpy as np
import time
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
import math

import create_datasets as cd

# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------

# ------------- Dataset Parameters -------------------

distance_from_sz = 2  # hours, minimum time between inter sample and sz
distance_from_inter = 2 # hours, minimum time between inter samples



class WholeDataset(Dataset):

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

        f = h5py.File('/media/NVdata/SzTimes/all_train_2step_%d_%d.mat' % (train_percent, iPt), 'r')
        self.labels = np.asarray(f['sample_labels'])
        self.times = np.asarray(f['sample_times'])
        f.close()


    def test(self):
        print(self.labels)

    # def __len__(self):
    #     return self.len

    def __getitem__(self, idx):  # DATA doesn't have to be loaded until here!

        y = self.labels[idx]
        if y==2 or y==-1:
            y=0

        start = self.times[idx]
        end = start + 600

        x = cd.get_data(self.iPt, start, end)

        return torch.tensor(x), torch.tensor(y)


class BalancedData(Dataset):

    def __init__(self, pt, train=True, train_percent=80, stepback=2, transform=None):
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

        print('Selecting interictal samples')
        inter_times = self.select_interictal()
        print('Complete')
        sz_times = self.times[self.labels==1]
        self.shuffle_samples(sz_times, inter_times)
        self.len = sz_times.size + inter_times.size

    def __len__(self):
        return self.len

    def select_interictal(self):
        num_sz = int(np.sum(self.labels[self.labels==1]))
        length = self.labels.size
        sz_times = self.times[self.labels==1]
        inter_times = np.zeros((sz_times.shape))

        for i in range(num_sz):
            found = False
            # select random index and see if its far enough away
            while not found:
                k = np.random.randint(0,length)
                time_to_szs = np.abs(sz_times - self.times[k])
                if i>0:
                    time_to_inter = np.abs(inter_times - self.times[k])
                    if time_to_szs.min()/3600 > distance_from_sz and time_to_inter.min()/3600 > distance_from_inter:
                        found=True
                else:
                    if time_to_szs.min()/3600 > distance_from_sz:
                        found=True

            inter_times[i] = self.times[k]

        return inter_times

    def shuffle_samples(self, sz_times, inter_times):

        times = np.concatenate((sz_times, inter_times))
        labels = np.concatenate((np.ones(sz_times.shape), np.zeros(inter_times.shape)))
        inds = np.arange(times.shape[0])
        np.random.shuffle(inds)

        self.select_times = times[inds]
        self.select_labels = labels[inds]

    def __getitem__(self, idx):  # DATA doesn't have to be loaded until here!

        y = self.select_labels[idx]
        if y == 2 or y == -1:
            y = 0

        start = self.select_times[idx]
        end = start + 600
        x_np = cd.get_data(self.pt, start, end)
        x_np[np.isnan(x_np)]=0
        # plt.plot(x_np[0])
        # plt.show()
        x = torch.tensor(x_np, dtype=torch.float32)
        # print('X size', x.size())
        if self.transform:
            # print(x.size)
            x = self.transform(x)

        return x, torch.tensor(y, dtype=torch.float32)


        # Randomly select interictal (class=0)
        # Check against requirements
            # 5 hrs from sz
            # Max dropout (0.5)
            # Min dropout (average / 4)
            # 2 hrs from any other interictal
        # Do so until matched to sz number


class BalancedData1m(Dataset):

    def __init__(self, pt, train=True, train_percent=80, stepback=2, transform=None):
        self.pt=pt
        self.train=train
        self.train_percent=train_percent
        self.stepback=stepback
        self.transform=transform
        self.balancedData = BalancedData(pt, train, train_percent, stepback, transform)
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

        x = sample[0][:,23977*k:23977*(k+1)]

        return x, y
