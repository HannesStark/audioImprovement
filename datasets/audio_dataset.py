import os
from typing import List, Dict, Any, Tuple

import librosa
import numpy as np
import warnings
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Dataset of noises from a directory.
    """

    def __init__(self, audio_dir: str, sample_rate=None, transform=None) -> None:
        """
        Args:
            audio_dir (string): Path to .wav, .mp3, .flac and other files of audios.
            sample_rate (int): SR to which the audio will be resampled. Using native SR if this is None.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.audio_names: List[str] = os.listdir(audio_dir)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        audio_name = self.audio_names[index]
        audio, sample_rate = librosa.load(path=os.path.join(self.audio_dir, audio_name), sr=self.sample_rate)

        if self.transform:
            audio = self.transform(audio)

        return audio, sample_rate

    def __len__(self) -> int:
        return len(self.audio_names)
