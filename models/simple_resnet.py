import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleResNet(nn.Module):

    def __init__(self):
        super(SimpleResNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv1d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv1d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv1d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv1d(32, 32, 3, 1, 1)
        self.conv6 = nn.Conv1d(32, 1, 3, 1, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        res1 = x
        x = self.conv2(x)
        x = F.selu(x)
        x = self.conv3(x)
        x += res1
        x = F.selu(x)
        res1 = x
        x = self.conv4(x)
        x = F.selu(x)
        x = self.conv5(x)
        x += res1
        x = F.selu(x)
        x = self.conv6(x)
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
