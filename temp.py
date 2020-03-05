import matplotlib.pyplot as plt
import os

import numpy as np
import soundfile as sf
import librosa
import torch
from torchsummary import summary

from datasets.from_segments_dir_dataset import FromSegmentsDirDataset
from datasets.audio_dataset import AudioDataset
from models.u_net import UNet
from utils import create_noisy_clip_dir, resample_directory, create_clean_noisy_pair_dirs, train_val_split
from datasets.transforms import ToTensor
from models.super_simple import SuperSimple
from solvers.solver import Solver



model = UNet()

summary(model, (1,20000))