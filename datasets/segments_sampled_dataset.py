import os
from typing import List, Any, Tuple

import numpy as np
from pydub import AudioSegment
import warnings
from torch.utils.data import Dataset


class SegmentsSampledDataset(Dataset):
    """Creates dataset of a specified size that is randomly sampled from the given speech directory.
    """

    def __init__(self, speech_dir: str, segment_length: int, dataset_length: int = None,
                 transform=None) -> None:
        """
        Args:
            speech_dir (string): Path to .wav, .mp3, .flac and other files of clean speech.
            segment_length (int): The length of a single noise and the corresponding clean segment.
            dataset_length (int): Specifies length of the dataset that will be randomly sampled from the specified
            speech directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.clip_names: List[str] = os.listdir(speech_dir)
        self.speech_dir = speech_dir
        self.segment_length = segment_length
        self.transform = transform
        self.dataset_length = dataset_length

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        clip = None
        while clip is None:
            clip_index = np.random.randint(0, self.dataset_length)
            clip_name = self.clip_names[clip_index]
            file_extension: str = os.path.splitext(clip_name)[1][1:]
            clip_candidate = AudioSegment.from_file(format=file_extension,
                                                    file=os.path.join(self.speech_dir, clip_name))
            if clip_candidate.frame_count() < self.segment_length:
                warnings.warn(
                    "The clip -" + clip_name + "- from the speech directory was smaller than the specified segment_length. It will be skipped.")
            else:
                clip = clip_candidate
        segment_index = np.random.randint(0, clip.frame_count() - self.segment_length)
        segment_index_in_millisec = 1000 * segment_index / clip.frame_rate
        segment_length_in_millisec = 1000 * self.segment_length / clip.frame_rate
        clean_segment = clip[segment_index_in_millisec: segment_index_in_millisec + segment_length_in_millisec]
        noisy_segment = clean_segment
        if self.transform:
            noisy_segment, clean_segment = self.transform((noisy_segment, clean_segment))

        return noisy_segment, clean_segment

    def __len__(self) -> int:
        return self.dataset_length
