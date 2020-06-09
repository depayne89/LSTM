import numpy as np
import torch, torchaudio, torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt

audiofile = '/home/daniel/Desktop/example.wav'

waveform, sample_rate = torchaudio.load(audiofile)
print(waveform.size)
nfft = int(waveform.shape[1]**.5*2)
print(nfft)

spec = torchaudio.transforms.Spectrogram(n_fft=nfft)(waveform)
# n_fft take double the square root to get a square

print(spec.shape)
# plt.figure()
# plt.imshow(spec[0,:,:].numpy())
# plt.show()
