import numpy as np
import torch, torchaudio, torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import os
import visualize as vis
import get_datasets as gd
import scipy
from scipy import signal
import create_datasets as cd
import metrics as m
import models

learning_rate = .001

# 1 min

min_c1 = 16  # 15
min_c2 = 32  # 64
min_c3 = 64
min_fc1 = 32  # 128
min_fc2 = 16

min_version = 0
trans_text = '_spec'
min_batch_size = 160  # for 1mBalanced: 20 samples / sz


min_model_name = '1m_v%d_c%dp4c%dp4d%dd_%d_adam' % (min_version, min_c1, min_c2, min_fc1, min_batch_size) + str(
        learning_rate)[2:] + trans_text

for pt in [1,3,6,8,9,10,11,13,15]:

    min_model_path = "/media/projects/daniel_lstm/models/" + min_model_name + "_%d.pt" % pt

    min_model = models.CNN1min_spec(out1=min_c1, out2=min_c2, out3=min_c3, out4=min_fc1, out5=min_fc2)
    torch.save(min_model, min_model_path)





