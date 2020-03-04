import matplotlib.pyplot as plt
import os

import numpy as np
import soundfile as sf
import librosa
import torch

from datasets.from_segments_dir_dataset import FromSegmentsDirDataset
from datasets.audio_dataset import AudioDataset
from datasets.data_utils import create_noisy_clip_dir, resample_directory, create_clean_noisy_pair_dirs, train_val_split

clean_dir = 'F:/datasets/libri_speech_subset_segments20000_clean'
noisy_dir = 'F:/datasets/libri_speech_subset_segments20000_noisy'

full_data = FromSegmentsDirDataset(noisy_dir, clean_dir)
train_data, val_data = train_val_split(full_data, 0.8)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=0)

