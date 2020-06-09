import numpy as np
import torch as t
import torchaudio


class Basic(t.nn.Module):
    def __init__(self):
        super(Basic, self).__init__()

        self.linear1 = nn.Linear()