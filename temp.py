import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as ColorNorm
import librosa
import numpy as np

import h5py
import torch
from torchsummary import summary
from torchvision import transforms

from datasets.audio_dataset import AudioDataset
from datasets.hdf5_dataset import HDF5Dataset
from datasets.segments_dataset import SegmentsDataset
from datasets.transforms import Normalize, NoiseTransform, ToTensor
from models.ae_middle_selu import AEMiddleSelu
from models.ae_simple import AESimple
from models.unet_undercomplete import UNetUndercomplete
from solvers.solver import Solver
from utils import create_hdf5, get_audio_list, train_val_split, create_noisy_clip_dir
import soundfile as sf
from scipy.signal import stft

audio, sr = librosa.load("F:/datasets/libri_speech_subset/19-198-0001.flac", sr=None)
audio = audio[:16384]
print(len(audio))
audio_tensor = torch.from_numpy(audio)
n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft)
# STFT
spectral = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, window=window)
print(spectral.shape)

real, imag = spectral[..., 0], spectral[..., 1]
print(real.shape)
mag = np.sqrt(real ** 2 + imag ** 2)
print(mag[0][50])


fig, ax0 = plt.subplots(nrows=1)

im = ax0.pcolormesh(mag)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()

phase = torch.atan2(imag, real)

logits_real = mag * torch.cos(phase)
logits_imag = mag * torch.sin(phase)
torch.istft