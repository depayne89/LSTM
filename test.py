import numpy as np
import torch, torchaudio, torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py

import preprocessing as pre
import visualize as vis



f = h5py.File('/media/NVdata/SzTimes/all_train_1step_50_1.mat', 'r')
sample_times = np.asarray(f['sample_times'])
print(sample_times.shape)
print(sample_times[:3])




# Displaying data

# pt =2
# start = 100100  # time since recording start, seconds
# end = 101000
#
# sample = pre.get_data(pt, start, end)
# print(sample.shape)
# vis.raw_eeg(pt, sample, start, end)

# vis.raw_sz_eeg(pt, 1, 20, 40, inter=False, inter_jump = 60*60*2, train=True)



