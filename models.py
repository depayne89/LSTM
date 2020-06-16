import numpy as np
import torch as t
import torchaudio


class CNN1min(t.nn.Module):
    def __init__(self, out1=32, out2=16):
        super(CNN1min, self).__init__()
        #16, 23977 (1 min)
        self.cnn1 = t.nn.Conv1d(in_channels=16, out_channels=out1, kernel_size=5, padding=0)
        #32, 23973
        self.maxpool1 = t.nn.MaxPool1d(kernel_size=4, stride=4)
        #32, 5993

        self.cnn2 = t.nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=5, padding=0)
        #16, 5989
        self.maxpool2 = t.nn.MaxPool1d(kernel_size=4, stride=4)
        #16, 1497

        self.fc = t.nn.Linear(out2*1497, 2)

    def forward(self, x):
        x = self.cnn1(x)
        x = t.relu(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = t.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
