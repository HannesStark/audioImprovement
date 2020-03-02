import torch

from datasets.audio_dataset import AudioDataset
from utils import clean_audio
import matplotlib.pyplot as plt

noisy_clips_dir = 'F:/datasets/libri_speech_subset_noisy'
data_path_noise = 'F:/datasets/Nonspeech'

noisy_clips_dataset = AudioDataset(noisy_clips_dir)
noisy_clip = noisy_clips_dataset[3]

model = torch.load('saved/firstTestNet20000')

cleaned_output = clean_audio(noisy_clip, model, 20000, batch_size=3)

plt.plot(noisy_clip.get_array_of_samples(), label='Noisy Input')
plt.plot(cleaned_output.get_array_of_samples(), label='Cleaned Output')
plt.legend(loc="upper right")
plt.show()

cleaned_output.export("audio/.output.wav", format="wav")
noisy_clip.export("audio/.input.wav", format="wav")
