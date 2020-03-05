from typing import Tuple, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import overlay_with_noise


class NoiseTransform():
    """Add noise to clean speech.

    Args:
        noise_dataset str: PyTorch Dataset with noises
    """

    def __init__(self, noise_dataset: Union[Dataset, List[Tuple[np.ndarray, int]]]):
        self.noise_dataset = noise_dataset
        self.number_of_noises = len(noise_dataset)

    def __call__(self, segments: Tuple[np.ndarray, np.ndarray, int]):
        noisy_segment, clean_segment, sample_rate = segments
        return overlay_with_noise(noisy_segment, sample_rate, self.noise_dataset), clean_segment, sample_rate


class ToTensor():
    """Add channels dimension if there is none and turn np.array into torch.Tensor."""

    def __init__(self):
        pass

    def __call__(self, segments: Union[Tuple[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]]):
        if len(segments) == 3:  # case of sample rate included in tuple
            noisy_segment, clean_segment, sample_rate = segments
            if noisy_segment.ndim == 1:
                noisy_segment = noisy_segment[np.newaxis, ...]
            if clean_segment.ndim == 1:
                clean_segment = clean_segment[np.newaxis, ...]

            return torch.from_numpy(noisy_segment).float(), torch.from_numpy(clean_segment).float(), sample_rate
        noisy_segment, clean_segment = segments
        if noisy_segment.ndim == 1:
            noisy_segment = noisy_segment[np.newaxis, ...]
        if clean_segment.ndim == 1:
            clean_segment = clean_segment[np.newaxis, ...]

        return torch.from_numpy(noisy_segment).float(), torch.from_numpy(clean_segment).float()


class Normalize():
    """Normalize to [-1,1]."""

    def __init__(self):
        pass

    def __call__(self, segments: Tuple[np.ndarray, np.ndarray, int]):
        noisy_segment, clean_segment, sample_rate = segments
        noisy_segment = noisy_segment / np.amax(np.abs(noisy_segment))
        clean_segment = clean_segment / np.amax(np.abs(clean_segment))
        return noisy_segment, clean_segment, sample_rate
