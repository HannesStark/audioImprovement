import os
import warnings
from typing import List, Tuple

import h5py
import librosa
import numpy as np
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Dataset of noisy and clean pairs taken from an hdf5 file
    """

    def __init__(self, hdf5_path: str, transform=None) -> None:
        """
        Args:
            hdf5_path (str): path to an hdf5 file with clean and noisy segments
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.data = h5py.File(hdf5_path, 'r')['data']
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        noisy_segment = self.data[index][0]
        clean_segment = self.data[index][1]
        if len(clean_segment) != len(noisy_segment):
            warnings.warn(
                'Length of noisy segment was not the same as length of clean segment for index: ' + str(index))
        if self.transform:
            noisy_segment, clean_segment = self.transform((noisy_segment, clean_segment))

        return noisy_segment, clean_segment

    def __len__(self) -> int:
        return self.data.shape[0]
