import torch

from datasets.audio_dataset import AudioDataset
import soundfile as sf
from utils import clean_audio
import matplotlib.pyplot as plt

noisy_clips_dir = 'F:/datasets/libri_speech_subset_noisy'
data_path_noise = 'F:/datasets/Nonspeech'

noisy_clips_dataset = AudioDataset(noisy_clips_dir)
noisy_clip, sample_rate = noisy_clips_dataset[7]

model_name = "AEStandardBigMiddle"
segment_length = 16384
model = torch.load('saved/' + model_name + str(segment_length) + '.model', map_location=torch.device('cpu'))

cleaned_output = clean_audio(noisy_clip, model, segment_length, batch_size=3)

plt.plot(noisy_clip, label='Noisy Input')
plt.plot(cleaned_output[0], label='Cleaned Output')
plt.legend(loc="upper right")
plt.show()

sf.write("audio/" + model_name + "_out.wav",cleaned_output[0], sample_rate)
sf.write("audio/" + model_name + "_in.wav", noisy_clip, sample_rate)
