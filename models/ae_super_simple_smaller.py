import torch
import torch.nn as nn
import torch.nn.functional as F


class AESuperSimpleSmaller(nn.Module):

    def __init__(self):
        super(AESuperSimpleSmaller, self).__init__()

        self.enc1 = nn.Conv1d(1, 1, 3, 1, 1)
        self.enc2 = nn.Conv1d(1, 1, 3, 2, 1)
        self.enc3 = nn.Conv1d(1, 1, 3, 2, 1)
        self.enc4 = nn.Conv1d(1, 1, 3, 2, 1)
        self.dec4 = nn.ConvTranspose1d(1, 1, 3, 2, 1, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(1, 1, 3, 2, 1, output_padding=1)
        self.dec2 = nn.ConvTranspose1d(1, 1, 3, 2, 1, output_padding=1)
        self.dec1 = nn.Conv1d(1, 1, 3, 1, 1)

    def forward(self, x):
        x = F.selu(self.enc1(x))
        x = F.selu(self.enc2(x))
        x = F.selu(self.enc3(x))
        x = F.selu(self.enc4(x))

        x = F.selu(self.dec4(x))
        x = F.selu(self.dec3(x))
        x = F.selu(self.dec2(x))
        x = self.dec1(x)
        return torch.tanh(x)

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
