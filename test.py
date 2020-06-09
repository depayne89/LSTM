import numpy as np
import torch, torchaudio, torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import time
import numpy as np

sz_times = np.array([10,11,12,13,14])
inter_times = np.array([20,21,22,23,24])

times = np.concatenate((sz_times, inter_times))
labels = np.concatenate((np.ones(sz_times.shape), np.zeros(inter_times.shape)))
inds = np.arange(times.shape[0])
np.random.shuffle(inds)

select_times = times[inds]
select_labels = labels[inds]

print(select_times, select_labels)
