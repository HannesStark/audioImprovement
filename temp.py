import matplotlib.pyplot as plt
import os

import numpy as np
import soundfile as sf
import librosa
import torch

from datasets.from_segments_dir_dataset import FromSegmentsDirDataset
from datasets.audio_dataset import AudioDataset
from datasets.data_utils import create_noisy_clip_dir, resample_directory, create_clean_noisy_pair_dirs, train_val_split
from datasets.transforms import ToTensor
from models.super_simple import SuperSimple
from solvers.solver import Solver

clean_dir = 'F:/datasets/libri_speech_subset_segments20000_clean'
noisy_dir = 'F:/datasets/libri_speech_subset_segments20000_noisy'

segment_length = 20000

full_data = FromSegmentsDirDataset(noisy_dir, clean_dir, transform=ToTensor())
train_data, val_data = train_val_split(full_data, 0.8)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, shuffle=False, num_workers=0)


model = SuperSimple()

solver = Solver(optim_args={"lr": 1e-2, "weight_decay": 0}, loss_func=torch.nn.MSELoss(), create_plots=False)
solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=30)

model.save("saved/FromColab" + str(segment_length) + '.model')

plt.plot(solver.train_loss_history, label='Train loss')
plt.plot(solver.val_loss_history, label='Val loss')
plt.legend(loc="upper right")
plt.show()

