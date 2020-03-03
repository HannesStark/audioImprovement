import os
from typing import List, Tuple

import librosa
import numpy as np
import warnings
from torch.utils.data import Dataset


class SegmentsSampledDataset(Dataset):
    """Creates dataset of a specified size that is randomly sampled from the given speech directory.
    """

    def __init__(self, speech_dir: str, segment_length: int, dataset_length: int, sample_rate=None,
                 transform=None) -> None:
        """
        Args:
            speech_dir (string): Path to .wav, .mp3, .flac and other files of clean speech.
            segment_length (int): The length of a single noise and the corresponding clean segment.
            dataset_length (int): Specifies length of the dataset that will be randomly sampled from the specified
            speech directory.
            sample_rate (int): SR to which the audio will be resampled. Using native SR if this is None.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.clip_names: List[str] = os.listdir(speech_dir)
        self.speech_dir = speech_dir
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.transform = transform
        self.dataset_length = dataset_length

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        clip = None
        while clip is None:
            clip_index = np.random.randint(0, self.dataset_length)
            clip_name = self.clip_names[clip_index]
            clip_candidate, sample_rate = librosa.load(path=os.path.join(self.speech_dir, clip_name),
                                                       sr=self.sample_rate)
            if len(clip_candidate) < self.segment_length:
                warnings.warn(
                    "The clip -" + clip_name + "- from the speech directory was smaller than the specified segment_length. It will be skipped.")
            else:
                clip = clip_candidate
        segment_index = np.random.randint(0, len(clip) - self.segment_length)
        clean_segment = clip[segment_index: segment_index + self.segment_length]
        noisy_segment = clean_segment
        if self.transform:
            noisy_segment, clean_segment = self.transform((noisy_segment, clean_segment, sample_rate))

        return noisy_segment, clean_segment

    def __len__(self) -> int:
        return self.dataset_length
