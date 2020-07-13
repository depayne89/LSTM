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

x = np.array([1,2,3])
m=1
for i in range(m-1):
    x = np.concatenate((x, np.array([1,2,3])))
print(x)



