import numpy as np
import torch
import torchaudio

import create_datasets as cd
import get_datasets as gd

dataset = gd.BalancedData(pt=1, transform=torchaudio.transforms.Spectrogram())
# shape: sample number, x or y, channel, time
print(dataset[0][0].size())

