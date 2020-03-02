import os
import warnings
from typing import List, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from torch.utils.data import Dataset
import torch


class TransformDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


def train_val_split(dataset: Dataset, train_size: float):
    """Create Train Val Split

    Args:
        dataset (Dataset): dataset to be split into train and val sets
        train_size (float): size of the train set relative to the total set
    """
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def overlay_with_noise(audio: AudioSegment, noise_dataset: Dataset) -> AudioSegment:
    """
        Randomly samples noises from noise_dataset and varies their volume and overlays them on the audio

    Args:
        audio (AudioSegment): a pydub audio clip to which the noise will be added
        noise_dataset (torch.utils.data.Dataset): Dataset with noises to randomly sample from
    """
    noise: AudioSegment = AudioSegment.empty()
    while audio.duration_seconds > noise.duration_seconds:
        noise_index = np.random.randint(0, len(noise_dataset))
        single_noise = noise_dataset[noise_index]
        single_noise = single_noise.set_frame_rate(audio.frame_rate)
        volume_difference = audio.dBFS - single_noise.max_dBFS
        noise += single_noise.apply_gain(np.random.rand() * volume_difference)
    volume_difference = audio.dBFS - noise.dBFS
    return audio.overlay(noise.apply_gain(volume_difference))


def create_noisy_clip_dir(input_dir: str, noise_dataset: Dataset, output_dir: str = None,
                          file_type: str = None) -> None:
    """
        Creates new directory and saves audios from input_dir overlayed with noises into the new directory.
        If the output_dir is None the output directory is created next to the input_dir with _noisy as suffix.
        Ignores .conf files in the input directory

    Args:
        input_dir (str): path to directory with audio files to add noise to
        noise_dataset (torch.utils.data.Dataset): Dataset with noises to randomly sample from
        output_dir (Optional[str]): path of the output directory. If this is None a directory next to input_dir is created
        file_type (Optional[str]): type of audio files. If None all files will be treated as audios except .conf
    """
    input_dir = os.path.split(input_dir + '/')[0]  # remove trailing / in case there is one
    if output_dir is None:
        output_dir = input_dir + '_noisy'
    if not os.path.exists(output_dir):  # create directory if it does not already exist
        os.mkdir(output_dir)
    audio_files = os.listdir(input_dir)
    audio_files_length = len(audio_files)
    for i, audio_file in enumerate(audio_files):
        output_file_path = os.path.join(output_dir, audio_file)
        print('Writing file ' + str(i + 1) + '/' + str(audio_files_length) + ' ' + output_file_path)
        file_extension: str = os.path.splitext(audio_file)[1][1:]
        if file_type is None:
            if file_extension != 'conf':
                audio = AudioSegment.from_file(format=file_extension, file=os.path.join(input_dir, audio_file))
                noisy_audio = overlay_with_noise(audio, noise_dataset)
                noisy_audio.export(output_file_path, format=file_extension)
        else:
            if file_extension == file_type:
                audio = AudioSegment.from_file(format=file_extension, file=os.path.join(input_dir, audio_file))
                noisy_audio = overlay_with_noise(audio, noise_dataset)
                noisy_audio.export(output_file_path, format=file_extension)


def audio_as_segments(audio: AudioSegment, segment_length: int) -> List[AudioSegment]:
    """
        Splits a single pydub AudioSegment into segments of length segment_length and returns a list of those segments.
        Always produces floor of (audio.frames / segment_length) + 1 many segments. So with 10 frames and
        segment_length of 2 you will get 6 segments where the last two are duplicates.

    Args:
        audio (AudioSegment): audio that will be splitted into segments of length segment_length
        segment_length (int): Length of the resulting segments
    """
    audio_size = audio.frame_count()
    if audio_size < segment_length:
        raise ValueError('Provided audio has less frames than segment_length')
    segments = []
    number_of_segments = int(audio_size / segment_length)
    segment_length_in_millisec = 1000 * segment_length / audio.frame_rate
    for number_of_segment in range(number_of_segments):
        segment_index_in_millisec = segment_length_in_millisec * number_of_segment
        segments.append(audio[segment_index_in_millisec: segment_index_in_millisec + segment_length_in_millisec])
    segments.append(audio[audio.duration_seconds * 1000 - segment_length_in_millisec:])
    return segments


def audio_segment_to_norm_tensor(audio: AudioSegment) -> Tuple[torch.Tensor, int]:
    """
        Normalizes an AudioSegment to a Tensor in range [-1,1] and adds a channel dimensionality if there is none.
        Returns the normalization constant and the normalized Tensor

    Args:
        audio (AudioSegment): audio that will be normalized
    """
    audio_array = np.array(audio.get_array_of_samples())
    max = np.amax(np.abs(audio_array))
    audio_array = audio_array / max
    if audio_array.ndim == 1:
        audio_array = audio_array[np.newaxis, ...]

    return torch.from_numpy(audio_array).float(), max
