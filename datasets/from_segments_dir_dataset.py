import os
from typing import List, Tuple

import librosa
import numpy as np
from torch.utils.data import Dataset


class FromSegmentsDirDataset(Dataset):
    """Dataset of noisy and clean pairs from two directories where the corresponding noisy and clean segment are in the
    same order in the directories i.e. they have the same name
    """

    def __init__(self, noisy_segemnts_dir: str, clean_segments_dir, transform=None) -> None:
        """
        Args:
            noisy_segemnts_dir (str): Path to .wav, .mp3, .flac and other files of noisy segments with same names as the clean segments
            clean_segments_dir (str): Path to .wav, .mp3, .flac and other files of clean segments with same names as the noisy segments
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.audio_names = os.listdir(clean_segments_dir)
        self.clean_segments_dir = clean_segments_dir
        self.noisy_segments_dir = noisy_segemnts_dir
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        audio_name = self.audio_names[index]
        clean_segment, sample_rate_clean = librosa.load(path=os.path.join(self.clean_segments_dir, audio_name), sr=None)
        noisy_segment, sample_rate_noisy = librosa.load(path=os.path.join(self.noisy_segments_dir, audio_name), sr=None)
        if sample_rate_clean != sample_rate_noisy:
            raise ValueError('Sample rates of noisy segment has to be the same as for the clean segment: ' + audio_name)
        if len(clean_segment) != len(noisy_segment):
            raise ValueError('Length of noisy segment has to be the same as for the clean segment: ' + audio_name)
        if self.transform:
            noisy_segment, clean_segment = self.transform((noisy_segment, clean_segment))

        return noisy_segment, clean_segment

    def __len__(self) -> int:
        return len(self.audio_names)
