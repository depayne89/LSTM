import torch
import numpy as np
import math

class weightedBCE(torch.nn.Module):

    def __init__(self, weights):
        super(weightedBCE, self).__init__()
        self.weights = weights

    def forward(self, x, y):
        loss = torch.Tensor(np.zeros(len(x)))
        for i, val in enumerate(x):
            if y[i]==1:
                loss[i] = -torch.log(val)*self.weights[1]
            else:
                loss[i] = -torch.log(1-val)*self.weights[0]

        return torch.sum(loss)/len(x)


