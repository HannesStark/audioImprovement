import soundfile as sf
import os
import warnings
from shutil import copy
from typing import List, Tuple, Union

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def train_val_split(dataset: Union[Dataset, List[Tuple[np.ndarray, int]]], train_size: float):
    """Create Train Val Split

    Args:
        dataset (Dataset): dataset to be split into train and val sets
        train_size (float): size of the train set relative to the total set
    """
    if isinstance(dataset, np.ndarray):
        train_indices, val_indices = disjoint_indices(len(dataset), train_size, random=True)
        return np.take(dataset, train_indices), np.take(dataset, val_indices)
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def get_audio_list(path: str, sample_rate=None) -> List[Tuple[np.ndarray, int]]:
    audios = []
    audio_files = os.listdir(path)
    for audio_file in audio_files:
        audios.append(librosa.load(path=os.path.join(path, audio_file), sr=sample_rate))
    return (audios)


def overlay_with_noise(audio: np.ndarray, audio_sample_rate: int,
                       noise_dataset: Union[Dataset, List[Tuple[np.ndarray, int]]]) -> np.ndarray:
    """
        Randomly samples noises from noise_dataset and varies their volume and overlays them on the audio. Audio should
        be normalized between [-1,1].

    Args:
        audio (np.array): audio clip to which the noise will be added
        noise_dataset (torch.utils.data.Dataset): Dataset or array with noises to randomly sample from
    """
    noise = []
    while len(noise) < len(audio):
        noise_index = np.random.randint(0, len(noise_dataset))
        single_noise, noise_sample_rate = noise_dataset[noise_index]
        if audio_sample_rate != noise_sample_rate:
            single_noise = librosa.resample(single_noise, orig_sr=noise_sample_rate, target_sr=audio_sample_rate)
        single_noise = single_noise / np.amax(np.abs(single_noise)) * np.random.rand() * 0.5
        noise = np.concatenate([noise, single_noise])
    noise = noise[:len(audio)]
    noisy_audio = audio + noise
    return noisy_audio * 0.5  # we can just divide by two here since both sounds are normalized to [-1,1]


def create_clean_noisy_pair_dirs(input_dir: str, noise_dataset: Dataset, segment_length: int,
                                 noisy_segments_dir: str = None, clean_segments_dir: str = None,
                                 file_type: str = None) -> None:
    """
        Splits audios from input dir into segments, NORMALIZES THEM and saves them to two directories where one contains
        clean segments and the other directory contains segments with the same name with added noise

    Args:
        input_dir (str): path to directory with audio files to add noise to
        noise_dataset (torch.utils.data.Dataset): Dataset with noises to randomly sample from
        noisy_segments_dir (Optional[str]): path of the output directory. If this is None a directory next to input_dir is created
        clean_segments_dir (Optional[str]): path of the output directory. If this is None a directory next to input_dir is created
        file_type (Optional[str]): type of audio files. If None all files will be treated as audios except .conf
    """
    input_dir = os.path.split(input_dir + '/')[0]  # remove trailing / in case there is one
    if noisy_segments_dir is None:
        noisy_segments_dir = input_dir + '_segments' + str(segment_length) + '_noisy'
    if clean_segments_dir is None:
        clean_segments_dir = input_dir + '_segments' + str(segment_length) + '_clean'
    if not os.path.exists(noisy_segments_dir):  # create directory if it does not already exist
        os.mkdir(noisy_segments_dir)
    if not os.path.exists(clean_segments_dir):  # create directory if it does not already exist
        os.mkdir(clean_segments_dir)
    audio_files = os.listdir(input_dir)
    for i, audio_file in enumerate(audio_files):
        print('Processing file ' + str(i + 1) + '/' + str(len(audio_files)))
        file_extension: str = os.path.splitext(audio_file)[1]
        audio_name: str = os.path.splitext(audio_file)[0]
        if file_type is None and file_extension != '.conf' or file_type is not None and file_extension == file_type:
            audio, sample_rate = librosa.load(path=os.path.join(input_dir, audio_file), sr=None)
            try:
                segments = audio_as_segments(audio, segment_length=segment_length)
            except ValueError:
                warnings.warn(
                    "The clip -" + audio_file + "- from the speech directory was smaller than the specified segment_length. It will be skipped.")
                break
            for j, clean_segment in enumerate(segments[:-1]):
                clean_segment = clean_segment / np.amax(np.abs(clean_segment))
                noisy_segment = overlay_with_noise(clean_segment, sample_rate, noise_dataset)
                sf.write(os.path.join(noisy_segments_dir, audio_name + '_' + str(j) + file_extension), noisy_segment,
                         sample_rate)
                sf.write(os.path.join(clean_segments_dir, audio_name + '_' + str(j) + file_extension), clean_segment,
                         sample_rate)


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
        file_extension: str = os.path.splitext(audio_file)[1]
        if file_type is None and file_extension != '.conf' or file_type is not None and file_extension == file_type:
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
        file_extension: str = os.path.splitext(audio_file)[1]
        if file_type is None and file_extension != '.conf' or file_type is not None and file_extension == file_type:
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

def clean_audio(audio: np.ndarray, model, segment_length: int, batch_size: int = 1) -> np.ndarray:
    """
        Splits an audio into segments and feeds the segments batch wise to the given model and stitches its outputs
        back together to then return it as an np.ndarray.

        Args:
            audio (np.ndarray): audio that should be denoised by the provided model
            model (torch.nn.Module): model to inference on segments from the provided audio
            segment_length (int): segment length that is expected by the provided model
            batch_size (int): size of batches processed by the model each iteration
    """
    segments: List[np.ndarray] = audio_as_segments(audio, segment_length)
    segments_tensors = []
    normalizing_factors = []
    for segment in segments:
        normalizing_factor = np.amax(np.abs(segment))
        segment = segment / normalizing_factor
        normalizing_factors.append(normalizing_factor)
        if segment.ndim == 1:
            segment = segment[np.newaxis, ...]
        segments_tensors.append(torch.from_numpy(segment).float())
    tensor_of_segments = torch.stack(segments_tensors)
    batches = torch.split(tensor_of_segments, batch_size)
    output = []
    for i, batch in enumerate(batches):
        print("Processing batch " + str(i + 1) + '/' + str(len(batches)))
        output.append(model(batch))
    output = torch.cat(output) # put into shape (number of segments, channels, frames in segment)
    output = output * torch.tensor(normalizing_factors)[:, None, None]  # multiply each segment with normalizing_factor
    output = output.view([output.shape[1], -1])  # put the segments back together
    cutout_index = 2 * output.shape[1] - segment_length - len(audio)
    output = torch.cat((output[:, :-segment_length], output[:, cutout_index:]), dim=1)  # cut out the overlap
    return output.detach().numpy()


def disjoint_indices(size: int, ratio: float, random=True) -> Tuple[np.ndarray, np.ndarray]:
    """
        Creates disjoint set of indices where all indices together are size many indices. The first set of the returned
        tuple has size*ratio many indices and the second one has size*(ratio-1) many indices.

        Args:
            audio (size): total number of indices returned. First and second array together
            ratio (float): relative sizes between the returned index arrays
            random (boolean): should the indices be randomly sampled
    """
    if random:
        train_indices = np.random.choice(np.arange(size), int(size * ratio), replace=False)
        val_indices = np.setdiff1d(np.arange(size), train_indices, assume_unique=True)
        return train_indices, val_indices

    indices = np.arange(size)
    split_index = int(size * ratio)
    return indices[:split_index], indices[split_index:]


def split_from_libri_download() -> None:
    data_source = 'F:/datasets/train-clean-100.tar/LibriSpeech/train-clean-100'
    data_destination = 'F:/datasets/libri_speech'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_source):
        for file in f:
            if '.flac' in file:
                files.append(os.path.join(r, file))

    print(len(files))
    count = 0
    for f in files:
        count += 1
        print(count)
        copy(f, data_destination)
        print(f)
