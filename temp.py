import librosa
import numpy as np

import h5py

from datasets.audio_dataset import AudioDataset
from datasets.hdf5_dataset import HDF5Dataset
from utils import create_hdf5
import soundfile as sf

#noise_dataset = AudioDataset('F:/datasets/Nonspeech_SR16000')
#create_hdf5('F:\datasets\libri_speech', noise_dataset, 20000)

#f = h5py.File('F:\datasets/libri_speech_subset.hdf5', 'r')
#sf.write("noisy.wav", f['data'][1][0], samplerate=16000)
#sf.write("clean.wav", f['data'][1][1], samplerate=16000)
#print(f['data'][1])

dataset = HDF5Dataset('F:\datasets/libri_speech_subset.hdf5')

print(dataset[1])