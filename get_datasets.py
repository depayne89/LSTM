import numpy as np
import time
import h5py
from torch.utils.data import Dataset, DataLoader

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

        return x, y


class BalancedData(Dataset):

    def __init__(self, pt, train=True, train_percent=80, stepback=2, transform=None):
        self.pt=pt
        self.train_percent = train_percent
        self.stepback = stepback
        self.transform = transform
        self.train = train
        if train:
            f = h5py.File('/media/NVdata/SzTimes/all_train_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
        else:
            f = h5py.File('/media/NVdata/SzTimes/all_test_%dstep_%d_%d.mat' % (stepback, train_percent, pt), 'r')
        self.train_start = f['train_start']
        self.train_end = f['train_end']
        self.horizon = f['pred_horizon']
        self.times = np.asarray(f['sample_times'])
        self.labels = np.asarray(f['sample_labels'])
        self.dropout = np.asarray(f['sample_dropout'])
        f.close()

        print('Selecting interictal sampels')
        inter_times = self.select_interictal()
        print('Complete')
        sz_times = self.times[self.labels==1]
        self.shuffle_samples(sz_times, inter_times)

    def time_to_nearest(self, labels):
        forwards = np.zeros(labels.size)
        backwards = np.zeros(labels.size)
        out = np.zeros(labels.size)
        j = 0
        k = 0

        for i in range(labels.size):
            if labels[i] == 1:
                j = 0
            else:
                j += 1
            forwards[i] = j

        for i in range(labels.size):
            if labels[-i - 1] == 1:
                k = 0
            else:
                k += 1
            backwards[-i - 1] = k

        for i in range(labels.size):
            out[i] = np.minimum(forwards[i], backwards[i])

        return out


    def select_interictal(self):

        num_sz = int(np.sum(self.labels[self.labels==1]))
        length = self.labels.size
        print(length)
        sz_times = self.times[self.labels==1]
        time_to_sz = self.time_to_nearest(self.labels)

        inter_times = np.zeros((sz_times.shape))
        inter_labels = np.zeros((self.labels.shape))

        for i in range(num_sz):

            time_to_inter = self.time_to_nearest(inter_labels)
            valid_samples = np.asarray(np.where(np.logical_and(time_to_inter>(distance_from_inter*6), time_to_sz>(distance_from_sz*6))))[0]
            rand = np.random.randint(0, valid_samples.size)
            rand_i = valid_samples[rand]
            inter_times[i] = self.times[rand_i]
            inter_labels[rand_i]=1

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

        x = cd.get_data(self.pt, start, end)

        return x, y


        # Randomly select interictal (class=0)
        # Check against requirements
            # 5 hrs from sz
            # Max dropout (0.5)
            # Min dropout (average / 4)
            # 2 hrs from any other interictal
        # Do so until matched to sz number


# tester = WholeDataset(1)
# print(tester[0][0].shape)
# test = BalancedData(1)
