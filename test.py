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

loader = DataLoader(dataset=gd.BalancedDataCombo(1, train=True), batch_size=1)

for x, y in loader:
    print('x shape ', x)



