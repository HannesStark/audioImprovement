import os
from shutil import copy
from typing import List

import numpy as np
import torch
from pydub import AudioSegment

from datasets.data_utils import audio_as_segments, audio_segment_to_norm_tensor


def clean_audio(audio: AudioSegment, model, segment_length: int, batch_size: int = 1) -> AudioSegment:
    """
        Splits an audio into segments and feeds the segments batch wise to the given model and stitches its outputs
        back together to then return it as an AudioSegment.

        Args:
            audio (AudioSegment): audio that should be denoised by the provided model
            model (torch.nn.Module): model to inference on segments from the provided audio
            segment_length (int): segment length that is expected by the provided model
            batch_size (int): size of batches processed by the model each iteration
    """
    segments: List[AudioSegment] = audio_as_segments(audio, segment_length)
    segments_tensors = []
    normalizing_factors = []
    for segment in segments:
        normalized_tensor, normalizing_factor = audio_segment_to_norm_tensor(segment)
        segments_tensors.append(normalized_tensor)
        normalizing_factors.append(normalizing_factor)
    tensor_of_segments = torch.stack(segments_tensors)
    batches = torch.split(tensor_of_segments, batch_size)
    output = []
    for i, batch in enumerate(batches):
        print("Processing batch " + str(i + 1) + '/' + str(len(batches)))
        output.append(model(batch))
    output = torch.cat(output)
    output = output * torch.tensor(normalizing_factors)[:, None, None]  # multiply each segment with normalizing_factor
    output = output.view([audio.channels, -1])  # put the segments back together
    cutout_index = 2 * output.shape[1] - segment_length - int(audio.frame_count())
    output = torch.cat((output[:, :-segment_length], output[:, cutout_index:]), dim=1)  # cut out the overlap
    if audio.channels == 1: # pydub expects a one dimensional array if there is only one channel
        output = output.view(-1)
    return AudioSegment(np.array(output.detach(), dtype='int16'), sample_width=audio.sample_width,
                        channels=audio.channels, frame_rate=audio.frame_rate)

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
