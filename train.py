import numpy as np
from torch.utils.data import SequentialSampler, SubsetRandomSampler
from torchvision import transforms

import matplotlib.pyplot as plt
from datasets.data_utils import train_val_split, TransformDataset
from datasets.audio_dataset import AudioDataset
from datasets.segments_dataset import SegmentsDataset
from datasets.transforms import NoiseTransform, Normalize
from models.super_simple import SuperSimple
from models.simple_resnet import SimpleResNet
from solvers.solver import Solver
import torch

data_path_speech = 'F:/datasets/libri_speech_subset'
data_path_noise = 'F:/datasets/Nonspeech'
segment_length = 20000

noise_dataset = AudioDataset(data_path_noise)
train_noise, val_noise = train_val_split(noise_dataset, 0.8)

full_dataset = SegmentsDataset(speech_dir=data_path_speech, segment_length=segment_length)
train_data, val_data = train_val_split(full_dataset, 0.8)

train_data = TransformDataset(train_data, transforms.Compose([NoiseTransform(train_noise), Normalize()]))
val_data = TransformDataset(val_data, transforms.Compose([NoiseTransform(val_noise), Normalize()]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=False, num_workers=1,
                                           sampler=SubsetRandomSampler(np.arange(0, 30)))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=1)

model = SuperSimple()

solver = Solver(optim_args={"lr": 1e-2, "weight_decay": 0}, loss_func=torch.nn.MSELoss(), create_plots=False)
solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=3)

model.save("saved/firstTestNet" + str(segment_length))

plt.plot(solver.train_loss_history, label='Train loss')
plt.plot(solver.val_loss_history, label='Val loss')
plt.legend(loc="upper right")
plt.show()
