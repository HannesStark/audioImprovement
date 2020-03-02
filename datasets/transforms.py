from typing import Tuple
from pydub import AudioSegment
from torch.utils.data import Dataset

from datasets.data_utils import overlay_with_noise, audio_segment_to_norm_tensor


class NoiseTransform(object):
    """Add noise to clean speech.

    Args:
        noise_dataset str: PyTorch Dataset with noises
    """

    def __init__(self, noise_dataset: Dataset):
        self.noise_dataset = noise_dataset
        self.number_of_noises = len(noise_dataset)

    def __call__(self, segments):
        noisy_segment, clean_segment = segments
        return overlay_with_noise(noisy_segment, self.noise_dataset), clean_segment


class Normalize(object):
    """Normalize AudioSegment to [-1,1] range numpy array."""

    def __init__(self):
        pass

    def __call__(self, segments: Tuple[AudioSegment, AudioSegment]):
        noisy_segment, clean_segment = segments
        noisy_segment, _ = audio_segment_to_norm_tensor(noisy_segment)
        clean_segment, _ = audio_segment_to_norm_tensor(clean_segment)
        return noisy_segment, clean_segment
