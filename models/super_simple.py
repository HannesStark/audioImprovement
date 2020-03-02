import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperSimple(nn.Module):

    def __init__(self):
        super(SuperSimple, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv1d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv1d(32, 1, 3, 1, 1)


    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = self.conv3(x)
        x = torch.tanh(x)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
