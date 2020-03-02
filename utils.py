import os
from shutil import copy
from typing import List, Tuple

import numpy as np
import torch

from datasets.data_utils import audio_as_segments


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
    output = torch.cat(output)
    output = output.view([audio.shape[0], -1])  # put the segments back together
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
    else:
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
