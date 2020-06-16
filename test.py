import numpy as np
import torch, torchaudio, torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

import validate as val

sz = np.array([.9, .6, .8, .6, .9, .7, .8, .7, 6])
inter = np.array([.5, .6, .4, .7, .5, .6, .4, .8, .3])

val.auc_hanleyci(sz, inter)
