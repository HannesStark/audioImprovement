import matplotlib.pyplot as plt
import librosa
import numpy as np

import h5py
from torchvision import transforms

from datasets.audio_dataset import AudioDataset
from datasets.hdf5_dataset import HDF5Dataset
from datasets.segments_dataset import SegmentsDataset
from datasets.transforms import Normalize, NoiseTransform, ToTensor
from utils import create_hdf5, get_audio_list, train_val_split
import soundfile as sf

noise_dataset = AudioDataset('F:/datasets/Nonspeech_SR16000')
create_hdf5('F:\datasets\libri_speech_subset', noise_dataset, 512)

# f = h5py.File('F:\datasets/libri_speech_subset.hdf5', 'r')
# sf.write("noisy.wav", f['data'][1][0], samplerate=16000)
# sf.write("clean.wav", f['data'][1][1], samplerate=16000)
# print(f['data'][1])

# dataset = HDF5Dataset('F:\datasets/libri_speech_subset.hdf5')
