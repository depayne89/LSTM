# import torch, torchaudio, torchvision
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import h5py
# import time
# import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib import rc
# import sys
# import os
import visualize as vis
import get_datasets as gd
# import scipy
# from scipy import signal
import create_datasets as cd
# import metrics as met
# import models
import h5py
import numpy as np
from datetime import datetime

import time

if False or False or True:
    print('True')
else:
    print('False')







""" TIMES
concatenating -                 537 (27.)
simple array -                  22.34 (.74)
reshaped array -                47.39 (1.7)
reshaped array (new var)        50.20 (1.7)
concat dict                     46.47 (1.6)



"""


