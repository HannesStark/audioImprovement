import os
from typing import List

import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
import torch


def train_val_split(dataset: Dataset, train_size: float):
    """Create Train Val Split

    Args:
        dataset (Dataset): dataset to be split into train and val sets
        train_size (float): size of the train set relative to the total set
    """
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def overlay_with_noise(audio: np.ndarray, audio_sample_rate: int, noise_dataset: Dataset) -> np.ndarray:
    """
        Randomly samples noises from noise_dataset and varies their volume and overlays them on the audio. Audio should
        be normalized between [-1,1].

    Args:
        audio (np.array): audio clip to which the noise will be added
        noise_dataset (torch.utils.data.Dataset): Dataset with noises to randomly sample from
    """
    noise = []
    while len(noise) < len(audio):
        noise_index = np.random.randint(0, len(noise_dataset))
        single_noise, noise_sample_rate = noise_dataset[noise_index]
        single_noise = librosa.resample(single_noise, orig_sr=noise_sample_rate, target_sr=audio_sample_rate)
        single_noise = single_noise / np.amax(np.abs(single_noise)) * np.random.rand() * 0.5
        noise = np.concatenate([noise, single_noise])
    noise = noise[:len(audio)]
    noisy_audio = audio + noise
    return noisy_audio / np.amax(np.abs(noisy_audio))


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
                audio, sample_rate = librosa.load(path=os.path.join(input_dir, audio_file), sr=None)
                noisy_audio = overlay_with_noise(audio, sample_rate, noise_dataset)
                sf.write(output_file_path, noisy_audio, sample_rate)
        else:
            if file_extension == file_type:
                audio, sample_rate = librosa.load(path=os.path.join(input_dir, audio_file), sr=None)
                noisy_audio = overlay_with_noise(audio, sample_rate, noise_dataset)
                sf.write(output_file_path, noisy_audio, sample_rate)


def resample_directory(input_dir: str, sample_rate: int, output_dir: str = None,
                       file_type: str = None) -> None:
    """
        Creates new directory and saves the resampled audios in there.

    Args:
        input_dir (str): path to directory with audio files to resample
        sample_rate (int): Target sample rate
        output_dir (Optional[str]): path of the output directory. If this is None a directory next to input_dir is created
        file_type (Optional[str]): type of audio files. If None all files will be treated as audios except .conf
    """
    input_dir = os.path.split(input_dir + '/')[0]  # remove trailing / in case there is one
    if output_dir is None:
        output_dir = input_dir + '_SR' + str(sample_rate)
    if not os.path.exists(output_dir):  # create directory if it does not already exist
        os.mkdir(output_dir)
    audio_files = os.listdir(input_dir)
    audio_files_length = len(audio_files)
    for i, audio_file in enumerate(audio_files):
        output_file_path = os.path.join(output_dir, audio_file)
        print('Resampling file ' + str(i + 1) + '/' + str(audio_files_length) + ' ' + output_file_path)
        file_extension: str = os.path.splitext(audio_file)[1][1:]
        if file_type is None:
            if file_extension != 'conf':
                audio, sample_rate = librosa.load(path=os.path.join(input_dir, audio_file), sr=sample_rate)
                sf.write(output_file_path, audio, sample_rate)
        else:
            if file_extension == file_type:
                audio, sample_rate = librosa.load(path=os.path.join(input_dir, audio_file), sr=sample_rate)
                sf.write(output_file_path, audio, sample_rate)


def audio_as_segments(audio: np.ndarray, segment_length: int) -> List[np.ndarray]:
    """
        Splits a single audio into segments of length segment_length and returns a list of those segments.
        Always produces floor of (audio.frames / segment_length) + 1 many segments. So with 10 frames and
        segment_length of 2 you will get 6 segments where the last two are duplicates.

    Args:
        audio (np.ndarray): audio that will be splitted into segments of length segment_length
        segment_length (int): Length of the resulting segments
    """
    audio_size = len(audio)
    if audio_size < segment_length:
        raise ValueError('Provided audio has less frames than segment_length')
    segments = []
    number_of_segments = int(audio_size / segment_length)
    for number_of_segment in range(number_of_segments):
        segment_index = segment_length * number_of_segment
        segments.append(audio[segment_index: segment_index + segment_length])
    segments.append(audio[audio_size - segment_length:])
    return segments
