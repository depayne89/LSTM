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

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
yhat = np.array([.0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
                 .1, .15, .2, .25, .3, .35, .4, .45, .5, .55])

print(m.sens_tiw(y, yhat, extrapolate=True))



