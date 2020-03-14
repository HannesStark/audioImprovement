import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoderSimple(nn.Module):

    def __init__(self):
        super(DenoisingAutoencoderSimple, self).__init__()

        self.enc1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.enc2 = nn.Conv1d(64, 64, 3, 2, 1)
        self.enc3 = nn.Conv1d(64, 64, 3, 2, 1)
        self.enc4 = nn.Conv1d(64, 128, 3, 2, 1)
        self.enc5 = nn.Conv1d(128, 128, 3, 2, 1)
        self.enc6 = nn.Conv1d(128, 128, 3, 2, 1)
        self.enc7 = nn.Conv1d(128, 128, 3, 2, 1)
        self.enc8 = nn.Conv1d(128, 128, 3, 2, 1)
        self.enc9 = nn.Conv1d(128, 128, 3, 2, 1)
        self.enc10 = nn.Conv1d(128, 128, 3, 2, 1)
        self.middle1 = nn.Conv1d(128, 128, 3, 1, 1)
        self.middle2 = nn.Conv1d(128, 128, 3, 1, 1)
        self.dec10 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.dec9 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.dec8 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.dec7 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.dec6 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.dec5 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.dec4 = nn.ConvTranspose1d(128, 64, 3, 2, 1, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(64, 64, 3, 2, 1, output_padding=1)
        self.dec2 = nn.ConvTranspose1d(64, 64, 3, 2, 1, output_padding=1)
        self.dec1 = nn.Conv1d(64, 1, 3, 1, 1)

    def forward(self, x):
        x = F.selu(self.enc1(x))
        x = F.selu(self.enc2(x))
        res2 = x
        x = F.selu(self.enc3(x))
        x = F.selu(self.enc4(x))
        res4 = x
        x = F.selu(self.enc5(x))
        x = F.selu(self.enc6(x))
        res6 = x
        x = F.selu(self.enc7(x))
        x = F.selu(self.enc8(x))
        res8 = x
        x = F.selu(self.enc9(x))
        x = F.selu(self.enc10(x))
        res10 = x

        x = F.selu(self.middle1(x))
        x = self.middle2(x)
        x += res10
        x = F.selu(x)

        x = F.selu(self.dec10(x))
        x = self.dec9(x)
        x += res8
        x = F.selu(x)
        x = F.selu(self.dec8(x))
        x = self.dec7(x)
        x += res6
        x = F.selu(x)
        x = F.selu(self.dec6(x))
        x = self.dec5(x)
        x += res4
        x = F.selu(x)
        x = F.selu(self.dec4(x))
        x = self.dec3(x)
        x += res2
        x = F.selu(x)
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
