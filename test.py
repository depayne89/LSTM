import numpy as np
import torch, torchaudio, torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import time

# import create_datasets as cd
#
# for pt in cd.pt_list():
#     f = h5py.File('/media/NVdata/Dropout/nanTen_patient%d.mat' % pt)
#     drop = np.asarray(f['dropout_ten']['train'])
#     print(pt, drop.mean())

# pt = 1
# error_t = 12878366.0  # and 600 after
# # start = error_t + 0
# # cd.get_data(pt, start, start + 600)
#
# r_start = cd.get_record_start(1)
#
# epoch = error_t + r_start
# date = time.gmtime(epoch) # 6/11/2010 at 9:00:00am
# print(date)

f = h5py.File('/media/NVdata/Patient_23_002/Data_2010_11_06/Hour_09/UTC_09_10_00.mat')
data = np.asarray(f['Data'])
print(data)
f.close()


