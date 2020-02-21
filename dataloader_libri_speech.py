import math
import os
import torch
from pydub import AudioSegment
import numpy as np
from typing import Optional, Callable, Union, List, Dict
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class NoisyCleanSpeechDataset(Dataset):
    """Dataset that maps clip of noisy speech to clip of clean Speech"""

    def __init__(self, speech_dir: str, noise_dir: str, transform=None) -> None:
        """
        Args:
            speech_dir (string): Path to .wav files of clean speech.
            noise_dir (string): Path to .wav files of noise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clip_names: List[str] = os.listdir(speech_dir)
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.clip_names)

    def __getitem__(self, index: int) -> Dict[str, any]:
        clip_name = self.clip_names[index]
        file_extension: str = os.path.splitext(clip_name)[1][1:]
        clip = AudioSegment.from_file(format=file_extension, file=os.path.join(self.speech_dir, clip_name))

        sample: Dict[str, any] = {'clean': clip, 'noisy': clip}

        if self.transform:
            sample = self.transform(sample)

        return sample


class NoiseTransform(object):
    """Add noise to clean speech.

    Args:
        noise_dir str: Directory with noise to be added to the noise_clip
    """

    def __init__(self, noise_dir: str):
        self.noise_dir = noise_dir

    def __call__(self, sample: Dict[str, AudioSegment]):
        clean_clip: AudioSegment = sample['clean']
        noisy_clip: AudioSegment = sample['noisy']

        noise_files: List[str] = os.listdir(self.noise_dir)
        noise: AudioSegment = AudioSegment.empty()
        while clean_clip.duration_seconds > noise.duration_seconds:
            noise_file: str = np.random.choice(noise_files)
            file_extension: str = os.path.splitext(noise_file)[1]
            single_noise = AudioSegment.from_file(format=file_extension, file=os.path.join(self.noise_dir, noise_file))
            volume_difference =  clean_clip.dBFS - single_noise.dBFS
            noise += single_noise.apply_gain(np.random.rand()*2*volume_difference)
        volume_difference = clean_clip.dBFS - noise.dBFS
        return {'clean': clean_clip, 'noisy': noisy_clip.overlay(noise.apply_gain(2*volume_difference))}


data_path_speech = 'F:/datasets/libri_speech'
data_path_noise = 'F:/datasets/Nonspeech'

noiseTransform: NoiseTransform = NoiseTransform(data_path_noise)
dataset: NoisyCleanSpeechDataset = NoisyCleanSpeechDataset(speech_dir=data_path_speech, noise_dir=data_path_noise,
                                                           transform=noiseTransform)
print(len(dataset))
sample: Dict[str, AudioSegment] = dataset[10000]
clean = sample['clean']
noisy = sample['noisy']

print(clean.max_dBFS)
print(noisy.max_dBFS)
plt.plot(noisy.get_array_of_samples())
plt.show()
clean.export("clean.wav", format="wav")
noisy.export("noisy.wav", format="wav")