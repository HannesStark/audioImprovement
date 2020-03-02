import os
import pickle
from typing import List, Dict, Any, Tuple

import soundfile
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
import warnings
from torch.utils.data import Dataset


class SegmentsDataset(Dataset):
    """Create a dataset from a speech directory. In initialization the directory is parsed and a mapping from indices
    to segments of the clips in the data is created. The size of that mapping (and the dataset) depends on the amount
    of provided data and on the segment_length.
    Upon initialization a config file is created in the speech_dir with a mapping from indices to segments
    """

    def __init__(self, speech_dir: str, segment_length: int, transform=None) -> None:
        """
        Args:
            speech_dir (string): Path to .wav, .mp3, .flac and other files of clean speech.
            segment_length (int): The length of a single noise and the corresponding clean segment.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.clip_names: List[str] = os.listdir(speech_dir)
        self.speech_dir = speech_dir
        self.segment_length = segment_length
        self.transform = transform

        index_segment_map_path = os.path.join(speech_dir, str(self.segment_length) + "_index_segment_map.conf")
        if os.path.isfile(index_segment_map_path):
            with open(index_segment_map_path, "rb") as fp:  # read index map if it already exists
                self.index_segment_map = pickle.load(fp)
            print("Using index to segment mapping found under " + index_segment_map_path)
        else:
            print("Start parsing directory to create dataset index to segment mapping")
            self.index_segment_map = []
            for clip_index, clip_name in enumerate(self.clip_names):  # All clips in directory
                file_name = os.path.join(self.speech_dir, clip_name)
                clip_size = soundfile.info(file_name).frames  # use soundfile here because it is MUCH faster than pydub
                segments_per_clip = int(clip_size / self.segment_length)
                if clip_size < self.segment_length:
                    warnings.warn(
                        "The clip -" + clip_name + "- from the speech directory was smaller than the specified segment_length. It will be skipped.")
                # All segments in clip. Notice that there will be no loop if segments_per_clip is 0.
                for segment_of_clip in range(segments_per_clip):
                    self.index_segment_map.append({'clip_name': clip_name, 'index': segment_of_clip})
            with open(index_segment_map_path, "wb") as fp:  # save index map as file
                pickle.dump(self.index_segment_map, fp)
            print("Finish parsing directory. Dataset length is: " + str(
                len(self.index_segment_map)) + ". Mapping is saved under " + index_segment_map_path)

        self.dataset_length = len(self.index_segment_map)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        segment_description = self.index_segment_map[index]
        file_extension: str = os.path.splitext(segment_description['clip_name'])[1][1:]
        clip = AudioSegment.from_file(format=file_extension,
                                      file=os.path.join(self.speech_dir, segment_description['clip_name']))
        segment_length_in_millisec = 1000 * self.segment_length / clip.frame_rate
        segment_index_in_millisec = segment_length_in_millisec * segment_description['index']
        clean_segment = clip[segment_index_in_millisec: segment_index_in_millisec + segment_length_in_millisec]

        noisy_segment = clean_segment
        if self.transform:
            noisy_segment, clean_segment = self.transform((noisy_segment, clean_segment))

        return noisy_segment, clean_segment

    def __len__(self) -> int:
        return self.dataset_length
