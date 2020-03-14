import numpy as np
from torch.utils.data import Subset
from torchvision import transforms

import matplotlib.pyplot as plt

from models.denoising_autoencoder_simple import DenoisingAutoencoderSimple
from models.u_net import UNet
from models.undercomplete_simple import UndercompleteSimple
from models.undercomplete_super_simple import UndercompleteSuperSimple
from utils import train_val_split, get_audio_list
from datasets.audio_dataset import AudioDataset
from datasets.segments_dataset import SegmentsDataset
from datasets.transforms import NoiseTransform, ToTensor, Normalize
from models.super_simple import SuperSimple
from models.simple_resnet import SimpleResNet
from solvers.solver import Solver
import torch

from utils import disjoint_indices

data_path_speech = 'F:/datasets/libri_speech_subset'
data_path_noise = 'F:/datasets/Nonspeech'
segment_length = 16384

noise_list = get_audio_list(data_path_noise, sample_rate=16000)
train_noise, val_noise = train_val_split(noise_list, 0.8)

audios_with_train_noise = SegmentsDataset(speech_dir=data_path_speech, segment_length=segment_length,
                                          transform=transforms.Compose(
                                              [Normalize(), NoiseTransform(train_noise), ToTensor()]))
audios_with_val_noise = SegmentsDataset(speech_dir=data_path_speech, segment_length=segment_length,
                                        transform=transforms.Compose(
                                            [Normalize(), NoiseTransform(val_noise), ToTensor()]))

audios_with_train_noise = Subset(audios_with_train_noise, np.arange(0,2000))
audios_with_val_noise = Subset(audios_with_val_noise, np.arange(0,2000))

train_indices, val_indices = disjoint_indices(len(audios_with_train_noise), 0.8, random=True)

train_data = Subset(audios_with_train_noise, train_indices)
val_data = Subset(audios_with_val_noise, val_indices)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, shuffle=False, num_workers=0)

model = UndercompleteSimple()

solver = Solver(optim_args={"lr": 1e-3, "weight_decay": 0}, loss_func=torch.nn.MSELoss(), create_plots=True)
solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=10)

model_name = "UNet"
model.save('saved/' + model_name + str(segment_length) + '.model')

plt.plot(solver.train_loss_history, label='Train loss')
plt.plot(solver.val_loss_history, label='Val loss')
plt.legend(loc="upper right")
plt.savefig('saved/' + model_name + str(segment_length))
plt.show()
