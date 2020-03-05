import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv1d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv1d(128, 256, 3, 2, 1)
        self.conv4 = nn.Conv1d(256, 512, 3, 2, 1)
        self.conv5 = nn.Conv1d(512, 512, 3, 2, 1)
        self.conv6 = nn.Conv1d(512, 512, 3, 2, 1)  # 625 with an input of 20,000
        self.conv7 = nn.Conv1d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv1d(512, 512, 3, 1, 1)
        self.convt1 = nn.ConvTranspose1d(512, 512, 3, 2, 1, output_padding=1)
        self.convt2 = nn.ConvTranspose1d(512, 512, 3, 2, 1, output_padding=1)
        self.convt3 = nn.ConvTranspose1d(512, 256, 3, 2, 1, output_padding=1)
        self.convt4 = nn.ConvTranspose1d(256, 128, 3, 2, 1, output_padding=1)
        self.convt5 = nn.ConvTranspose1d(128, 64, 3, 2, 1, output_padding=1)
        self.conv_final = nn.Conv1d(64, 1, 3, 1, 1)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        res1 = x
        x = F.selu(self.conv2(x))
        res2 = x
        x = F.selu(self.conv3(x))
        res3 = x
        x = F.selu(self.conv4(x))
        res4 = x
        x = F.selu(self.conv5(x))
        res5 = x
        x = F.selu(self.conv6(x))
        res6 = x

        x = F.selu(self.conv7(x))
        x = self.conv8(x)
        x += res6
        x = F.selu(x)

        x = self.convt1(x)
        x += res5
        x = F.selu(x)
        x = self.convt2(x)
        x += res4
        x = F.selu(x)
        x = self.convt3(x)
        x += res3
        x = F.selu(x)
        x = self.convt4(x)
        x += res2
        x = F.selu(x)
        x = self.convt5(x)
        x += res1
        x = F.selu(x)
        x = self.conv_final(x)
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
