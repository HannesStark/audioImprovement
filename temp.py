import matplotlib.pyplot as plt
import os

import numpy as np
import soundfile as sf
import librosa
from datasets.audio_dataset import AudioDataset
from datasets.data_utils import create_noisy_clip_dir, resample_directory

input_dir = 'F:/datasets/libri_speech_subset'
data_path_noise = 'F:/datasets/Nonspeech_SR16000'

noise_dataset = AudioDataset(data_path_noise)

instance, sr = noise_dataset[8]
sf.write("input.wav", instance, sr)
instance2 = instance / np.amax(np.abs(instance))

plt.plot(instance2)
plt.plot(instance)
plt.show()
sf.write("output.wav", instance2, sr)
