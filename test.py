import torch

from datasets.audio_dataset import AudioDataset
import soundfile as sf
from utils import clean_audio
import matplotlib.pyplot as plt

noisy_clips_dir = 'F:/datasets/libri_speech_subset_noisy'
data_path_noise = 'F:/datasets/Nonspeech'

noisy_clips_dataset = AudioDataset(noisy_clips_dir)
noisy_clip, sample_rate = noisy_clips_dataset[3]

model = torch.load('saved/firstTestNet20000')

cleaned_output = clean_audio(noisy_clip, model, 20000, batch_size=3)

plt.plot(noisy_clip, label='Noisy Input')
plt.plot(cleaned_output, label='Cleaned Output')
plt.legend(loc="upper right")
plt.show()

sf.write("audio/.output.wav",cleaned_output, sample_rate)
sf.write("audio/.input.wav", noisy_clip, sample_rate)
