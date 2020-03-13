import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoderSimple(nn.Module):

    def __init__(self):
        super(DenoisingAutoencoderSimple, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv1d(64, 64, 3, 2, 1)
        self.conv3 = nn.Conv1d(64, 64, 3, 2, 1)
        self.conv4 = nn.Conv1d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv1d(128, 128, 3, 2, 1)
        self.conv6 = nn.Conv1d(128, 128, 3, 2, 1)
        self.conv7 = nn.Conv1d(128, 128, 3, 2, 1)
        self.conv8 = nn.Conv1d(128, 128, 3, 2, 1)
        self.conv9 = nn.Conv1d(128, 128, 3, 2, 1)
        self.conv10 = nn.Conv1d(128, 128, 3, 2, 1)
        self.conv11 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.conv12 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.conv13 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.conv14 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.conv15 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.conv16 = nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1)
        self.conv17 = nn.ConvTranspose1d(128, 64, 3, 2, 1, output_padding=1)
        self.conv18 = nn.ConvTranspose1d(64, 64, 3, 2, 1, output_padding=1)
        self.conv19 = nn.ConvTranspose1d(64, 64, 3, 2, 1, output_padding=1)
        self.conv20 = nn.Conv1d(64, 1, 3, 1, 1)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))
        x = F.selu(self.conv7(x))
        x = F.selu(self.conv8(x))
        x = F.selu(self.conv9(x))
        x = F.selu(self.conv10(x))
        x = F.selu(self.conv11(x))
        x = F.selu(self.conv12(x))
        x = F.selu(self.conv13(x))
        x = F.selu(self.conv14(x))
        x = F.selu(self.conv15(x))
        x = F.selu(self.conv16(x))
        x = F.selu(self.conv17(x))
        x = F.selu(self.conv18(x))
        x = F.selu(self.conv19(x))
        x = self.conv20(x)
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
