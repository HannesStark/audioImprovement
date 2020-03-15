import matplotlib.pyplot as plt
import librosa
import numpy as np

import h5py
import torch
from torchsummary import summary
from torchvision import transforms

from datasets.audio_dataset import AudioDataset
from datasets.hdf5_dataset import HDF5Dataset
from datasets.segments_dataset import SegmentsDataset
from datasets.transforms import Normalize, NoiseTransform, ToTensor
from models.ae_middle_selu import AEMiddleSelu
from models.ae_simple import AESimple
from solvers.solver import Solver
from utils import create_hdf5, get_audio_list, train_val_split
import soundfile as sf

model = AEMiddleSelu()

summary(model, (1,16384))