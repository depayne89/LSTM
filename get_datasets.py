import numpy as np
import time
import h5py
from torch.utils.data import Dataset, DataLoader

# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------
# ------------------------------  NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!1 -----------------------------


import create_datasets as cd


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

# tester = WholeDataset(1)
# print(tester[0][0].shape)
