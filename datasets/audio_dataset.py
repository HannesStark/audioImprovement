import os
from typing import List, Dict, Any, Tuple

import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
import warnings
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Dataset of noises from a directory.
    """

    def __init__(self, audio_dir: str, transform=None) -> None:
        """
        Args:
            audio_dir (string): Path to .wav, .mp3, .flac and other files of audios.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.audio_names: List[str] = os.listdir(audio_dir)
        self.audio_dir = audio_dir
        self.transform = transform

    def __getitem__(self, index: int) -> AudioSegment:
        audio_name = self.audio_names[index]
        file_extension: str = os.path.splitext(audio_name)[1][1:]
        audio = AudioSegment.from_file(format=file_extension, file=os.path.join(self.audio_dir, audio_name))

        if self.transform:
            audio = self.transform(audio)

        return audio

    def __len__(self) -> int:
        return len(self.audio_names)
